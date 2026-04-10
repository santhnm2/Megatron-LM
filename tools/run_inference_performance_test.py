# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import asyncio
import logging
import random
import sys
import time
from typing import List

import torch

from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    start_text_gen_server,
    stop_text_gen_server,
)
from megatron.core.utils import trace_async_exceptions
from megatron.inference.utils import add_inference_args, get_dynamic_inference_engine
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron

logger = logging.getLogger(__name__)


def add_inference_benchmarking_args(parser):
    """Inference benchmarking arguments."""
    parser = add_inference_args(parser)

    group = parser.add_argument_group(title='inference_benchmarking')

    group.add_argument(
        "--num-input-tokens", type=int, default=128, help="Number of input tokens per request"
    )
    group.add_argument(
        "--benchmark-profile", action="store_true", default=False, help="If set, profile"
    )
    group.add_argument(
        "--port", type=int, default=5000, help="Port for the text generation server"
    )
    group.add_argument(
        "--host", type=str, default=None,
        help="Hostname or IP address to bind the server to. Defaults to 0.0.0.0 (all interfaces)."
    )
    group.add_argument(
        "--num-warmup-iterations", type=int, default=3, help="Number of warmup iterations"
    )
    return parser


def get_random_prompt_tokens(tokenizer, num_input_tokens) -> List[int]:
    """Generate random prompt tokens, excluding special tokens."""
    special_token_ids = set()
    try:
        if hasattr(tokenizer, 'bos') and tokenizer.bos is not None:
            special_token_ids.add(tokenizer.bos)
        if hasattr(tokenizer, 'eos') and tokenizer.eos is not None:
            special_token_ids.add(tokenizer.eos)
        if hasattr(tokenizer, 'eod') and tokenizer.eod is not None:
            special_token_ids.add(tokenizer.eod)
        if (
            hasattr(tokenizer, 'additional_special_tokens_ids')
            and tokenizer.additional_special_tokens_ids
        ):
            special_token_ids.update(tokenizer.additional_special_tokens_ids)
    except NotImplementedError:
        pass

    valid_token_ids = [i for i in range(tokenizer.vocab_size) if i not in special_token_ids]
    prompt_tokens = random.choices(valid_token_ids, k=num_input_tokens)
    assert len(prompt_tokens) == num_input_tokens
    return prompt_tokens


@trace_async_exceptions
async def run_benchmark_client(
    coordinator_addr: str,
    requests: List[InferenceRequest],
    sampling_params: SamplingParams,
    warmup_sampling_params: SamplingParams,
    num_warmup_iterations: int,
    benchmark_profile: bool,
):
    """Submit benchmark requests to the inference coordinator via ZMQ client.

    Runs as an async task alongside the engine loop. The client submits
    requests to the coordinator, which routes them to the engine running
    in a separate async task — eliminating the single-process contention.
    """
    client = InferenceClient(coordinator_addr, deserialize=False)
    client.start()
    logger.info("Benchmark client connected to coordinator.")

    try:
        # Warmup
        for i in range(num_warmup_iterations):
            print(f"Running warmup iteration {i + 1}...")
            warmup_future = client.add_request("warmup", warmup_sampling_params)
            await warmup_future

        if benchmark_profile:
            torch.cuda.cudart().cudaProfilerStart()

        # Submit all requests
        start_time = time.perf_counter()
        futures = []
        for req in requests:
            future = client.add_request(req.prompt_tokens, sampling_params)
            futures.append(future)

        results = await asyncio.gather(*futures)
        end_time = time.perf_counter()
        latency = end_time - start_time

        if benchmark_profile:
            torch.cuda.cudart().cudaProfilerStop()

        memory_allocated = torch.cuda.max_memory_allocated()

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            for idx, result in enumerate(results):
                print(f' \n------------- RESULT FOR PROMPT {idx} --------------- ')
                num_output_tokens = result.get('generated_length', 0)
                result_dict = {
                    'id': idx,
                    'num_input_tokens': len(requests[idx].prompt_tokens),
                    'num_output_tokens': num_output_tokens,
                    'latency': result.get('latency', latency),
                    'memory_usage_GB': memory_allocated / (1024**3),
                }
                print(result_dict)

            total_output_tokens = sampling_params.num_tokens_to_generate * len(requests)
            throughput = total_output_tokens / latency
            print(f"\nTotal latency: {latency:.3f}s")
            print(f"Throughput: {throughput:.1f} output tokens / second")

    finally:
        client.stop()


@trace_async_exceptions
async def run_benchmark(
    engine: DynamicInferenceEngine,
    requests: List[InferenceRequest],
    sampling_params: SamplingParams,
    warmup_sampling_params: SamplingParams,
    num_warmup_iterations: int,
    benchmark_profile: bool,
    server_port: int,
    hostname: str | None = None,
):
    """Start the coordinator and engine loop, then run the benchmark client."""

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    coordinator_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=None,
        launch_inference_coordinator=True,
        hostname=hostname,
    )

    try:
        if rank == 0:
            start_text_gen_server(
                coordinator_addr=coordinator_addr,
                tokenizer=engine.controller.tokenizer,
                rank=rank,
                server_port=server_port,
                hostname=hostname,
            )

        if rank == 0:
            # Run the benchmark client alongside the engine loop
            benchmark_task = asyncio.ensure_future(
                run_benchmark_client(
                    coordinator_addr=coordinator_addr,
                    requests=requests,
                    sampling_params=sampling_params,
                    warmup_sampling_params=warmup_sampling_params,
                    num_warmup_iterations=num_warmup_iterations,
                    benchmark_profile=benchmark_profile,
                )
            )
            # Wait for benchmark to finish, then cancel the engine loop
            await benchmark_task
        else:
            # Non-rank-0 workers just run the engine loop until cancelled
            await engine.engine_loop_task

    finally:
        if rank == 0:
            stop_text_gen_server()


@torch.inference_mode()
def main():
    """Main program."""
    initialize_megatron(
        extra_args_provider=add_inference_benchmarking_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'micro_batch_size': 1,
            'exit_on_missing_checkpoint': True,
        },
    )

    args = get_args()

    engine = get_dynamic_inference_engine()
    tokenizer = engine.controller.tokenizer

    assert (args.prompts is None) ^ (
        args.num_input_tokens is None
    ), "Exactly one of `--prompts` and `--num-input-tokens` must be specified"

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        top_n_logprobs=args.top_n_logprobs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        termination_id=-1,
    )
    sampling_params.add_attributes({"no_early_termination": True})

    warmup_sampling_params = SamplingParams(num_tokens_to_generate=10, termination_id=-1)

    requests = []
    batch_size = args.inference_dynamic_batching_max_requests or 1
    if args.num_input_tokens is not None:
        assert args.prompts is None
        for i in range(batch_size):
            prompt_tokens = get_random_prompt_tokens(tokenizer, args.num_input_tokens)
            requests.append(
                InferenceRequest(
                    request_id=str(time.monotonic()),
                    prompt=tokenizer.detokenize(prompt_tokens),
                    prompt_tokens=prompt_tokens,
                    inference_parameters=sampling_params,
                )
            )
    else:
        assert args.prompts is not None
        for prompt in args.prompts:
            requests.append(
                InferenceRequest(
                    request_id=str(time.monotonic()),
                    prompt=prompt,
                    prompt_tokens=tokenizer.tokenize(prompt),
                    inference_parameters=sampling_params,
                )
            )

    try:
        asyncio.run(
            run_benchmark(
                engine=engine,
                requests=requests,
                sampling_params=sampling_params,
                warmup_sampling_params=warmup_sampling_params,
                num_warmup_iterations=args.num_warmup_iterations,
                benchmark_profile=args.benchmark_profile,
                server_port=args.port,
                hostname=args.host,
            )
        )
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
