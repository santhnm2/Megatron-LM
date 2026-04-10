# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import multiprocessing as mp
import random
import time
from typing import List, Optional

import torch

from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams
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
        "--num-warmup-iterations", type=int, default=3, help="Number of warmup iterations"
    )
    return parser


def get_random_prompt_tokens(vocab_size, special_token_ids, num_input_tokens) -> List[int]:
    """Generate random prompt tokens, excluding special tokens."""
    valid_token_ids = [i for i in range(vocab_size) if i not in special_token_ids]
    prompt_tokens = random.choices(valid_token_ids, k=num_input_tokens)
    assert len(prompt_tokens) == num_input_tokens
    return prompt_tokens


def _get_special_token_ids(tokenizer) -> set:
    """Extract special token IDs from the tokenizer."""
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
    return special_token_ids


# ---------------------------------------------------------------------------
# Benchmark client — runs in a separate process
# ---------------------------------------------------------------------------


def _benchmark_client_worker(
    coordinator_addr: str,
    prompt_token_lists: List[List[int]],
    sampling_params_dict: dict,
    warmup_sampling_params_dict: dict,
    num_warmup_iterations: int,
    benchmark_profile: bool,
):
    """Synchronous entry point for the benchmark client subprocess.

    Spins up its own event loop and runs the async benchmark client.
    This process has no GPU context — it only talks to the coordinator
    over ZMQ, so there is zero contention with the engine process.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _async_benchmark_client(
                coordinator_addr=coordinator_addr,
                prompt_token_lists=prompt_token_lists,
                sampling_params=SamplingParams.deserialize(sampling_params_dict),
                warmup_sampling_params=SamplingParams.deserialize(warmup_sampling_params_dict),
                num_warmup_iterations=num_warmup_iterations,
                benchmark_profile=benchmark_profile,
            )
        )
    except KeyboardInterrupt:
        logger.info("Benchmark client interrupted.")
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


async def _async_benchmark_client(
    coordinator_addr: str,
    prompt_token_lists: List[List[int]],
    sampling_params: SamplingParams,
    warmup_sampling_params: SamplingParams,
    num_warmup_iterations: int,
    benchmark_profile: bool,
):
    """Submit benchmark requests to the coordinator and report results."""
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
        for prompt_tokens in prompt_token_lists:
            future = client.add_request(prompt_tokens, sampling_params)
            futures.append(future)

        results = await asyncio.gather(*futures)
        end_time = time.perf_counter()
        latency = end_time - start_time

        if benchmark_profile:
            torch.cuda.cudart().cudaProfilerStop()

        for idx, result in enumerate(results):
            print(f' \n------------- RESULT FOR PROMPT {idx} --------------- ')
            result_dict = {
                'id': idx,
                'num_input_tokens': len(prompt_token_lists[idx]),
                'num_output_tokens': result.get('generated_length', 0),
                'latency': result.get('latency', latency),
            }
            print(result_dict)

        total_output_tokens = sampling_params.num_tokens_to_generate * len(prompt_token_lists)
        throughput = total_output_tokens / latency
        print(f"\nTotal latency: {latency:.3f}s")
        print(f"Throughput: {throughput:.1f} output tokens / second")

    finally:
        client.stop()


# ---------------------------------------------------------------------------
# Engine server — runs in the main process (needs GPU + torch.distributed)
# ---------------------------------------------------------------------------


@trace_async_exceptions
async def _run_engine_with_benchmark(
    engine: DynamicInferenceEngine,
    prompt_token_lists: List[List[int]],
    sampling_params: SamplingParams,
    warmup_sampling_params: SamplingParams,
    num_warmup_iterations: int,
    benchmark_profile: bool,
    hostname: Optional[str] = None,
):
    """Start the coordinator and engine loop, then spawn the benchmark client process."""

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    coordinator_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=None,
        launch_inference_coordinator=True,
        hostname=hostname,
    )

    client_process = None
    try:
        if rank == 0:
            client_process = mp.Process(
                target=_benchmark_client_worker,
                args=(
                    coordinator_addr,
                    prompt_token_lists,
                    sampling_params.serialize(),
                    warmup_sampling_params.serialize(),
                    num_warmup_iterations,
                    benchmark_profile,
                ),
                daemon=True,
            )
            client_process.start()
            logger.info(f"Started benchmark client process (PID: {client_process.pid})")

        # Run the engine loop until the client process finishes
        if rank == 0:
            while client_process.is_alive():
                # Give the engine loop time to process requests
                await asyncio.sleep(0.1)
        else:
            await engine.engine_loop_task

    finally:
        if client_process is not None:
            if client_process.is_alive():
                client_process.terminate()
                client_process.join(timeout=5)
                if client_process.is_alive():
                    client_process.kill()
                    client_process.join()


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

    # Build prompt token lists to pass to the client process
    special_token_ids = _get_special_token_ids(tokenizer)
    prompt_token_lists = []
    batch_size = args.inference_dynamic_batching_max_requests or 1
    if args.num_input_tokens is not None:
        assert args.prompts is None
        for _ in range(batch_size):
            prompt_token_lists.append(
                get_random_prompt_tokens(tokenizer.vocab_size, special_token_ids, args.num_input_tokens)
            )
    else:
        assert args.prompts is not None
        for prompt in args.prompts:
            prompt_token_lists.append(tokenizer.tokenize(prompt))

    try:
        asyncio.run(
            _run_engine_with_benchmark(
                engine=engine,
                prompt_token_lists=prompt_token_lists,
                sampling_params=sampling_params,
                warmup_sampling_params=warmup_sampling_params,
                num_warmup_iterations=args.num_warmup_iterations,
                benchmark_profile=args.benchmark_profile,
                hostname=getattr(args, 'host', None),
            )
        )
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
