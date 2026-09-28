# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Strict replay of recorded Mamba prefill calls against pinned vLLM.

Requires both engine packages in the same CUDA Python environment and the
conv/scan capture format from the layer audit. The caller supplies capture
paths; no model weights or private prompts are distributed with this script.
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import torch
import triton

PREFIX = {"minf": "megatron.core.ssm.ops", "vllm": "vllm.model_executor.layers.mamba.ops"}


def read(path, preserve_strides=True):
    meta = json.loads(path.read_text())
    tensors = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
    result = {}
    for name, value in tensors.items():
        if preserve_strides:
            layout = meta['tensors'][name]
            restored = torch.empty_strided(
                layout['shape'], layout['stride'], dtype=value.dtype, device='cuda'
            )
            restored.copy_(value)
            result[name] = restored
        else:
            result[name] = value.cuda()
    return meta, result


def paired(path):
    candidates = list(
        path.parent.glob(
            path.name[:4] + '*_' + path.name.split('_', 2)[-1].replace('.input.', '.output.')
        )
    )
    assert len(candidates) == 1, candidates
    return candidates[0]


def pin(configs, remap=None):
    for entry in configs:
        module = remap if remap is not None else importlib.import_module(entry['module'])
        kernel = getattr(module, entry['attribute'])
        config = triton.Config(
            entry['kwargs'],
            num_warps=entry['num_warps'],
            num_stages=entry['num_stages'],
            num_ctas=entry['num_ctas'],
            maxnreg=entry['maxnreg'],
        )
        kernel.configs = [config]
        kernel.cache.clear()


def functions(engine):
    conv_name = 'causal_conv1d_varlen' if engine == 'minf' else 'causal_conv1d'
    conv = importlib.import_module(PREFIX[engine] + '.' + conv_name)
    scan = importlib.import_module(PREFIX[engine] + '.ssd_combined')
    return (
        conv.causal_conv1d_varlen_fn if engine == 'minf' else conv.causal_conv1d_fn,
        scan.mamba_chunk_scan_combined_varlen,
    )


def native(engine, kind, meta, tensors, expected_meta):
    function = functions(engine)[0 if kind == 'conv' else 1]
    allowed = inspect.signature(function).parameters
    kwargs = {k: v for k, v in meta['scalars'].items() if k in allowed}
    kwargs.update({k: v for k, v in tensors.items() if k in allowed})
    if kwargs.get('state_dtype'):
        kwargs['state_dtype'] = getattr(torch, kwargs['state_dtype'].split('.')[-1])
    if 'dt_limit' in kwargs:
        kwargs['dt_limit'] = tuple(kwargs['dt_limit'])
    state = None
    if kind == 'conv' and engine == 'vllm':
        selected = tensors['conv_states_selected']
        state = torch.empty_strided(
            (2, *selected.shape[1:]),
            meta['scalars']['original_conv_cache_stride'],
            dtype=selected.dtype,
            device='cuda',
        )
        state.zero_()
        state[1].copy_(selected[0])
        kwargs.update(
            conv_states=state,
            cache_indices=torch.ones_like(tensors['cache_indices']),
            metadata=None,
        )
    if kind == 'scan':
        layout = expected_meta['tensors']['out']
        kwargs['out'] = torch.empty_strided(
            layout['shape'],
            layout['stride'],
            dtype=getattr(torch, layout['dtype'].split('.')[-1]),
            device='cuda',
        )
        kwargs['out'].zero_()
    returned = function(**kwargs)
    outputs = (
        {'return': returned}
        if isinstance(returned, torch.Tensor)
        else {f'return_{i}': x for i, x in enumerate(returned) if isinstance(x, torch.Tensor)}
    )
    if kind == 'scan':
        outputs['out'] = kwargs['out']
    if state is not None:
        outputs['conv_states_selected_after'] = state[1:2]
    return outputs


def conv_call(engine, x, weight, bias, initial, function=None):
    function = function or functions(engine)[0]
    n, channels = x.shape
    cu = torch.tensor([0, n], dtype=torch.int32, device='cuda')
    if engine == 'minf':
        return function(
            x.contiguous(), weight.contiguous(), bias, cu, initial_states=initial, activation='silu'
        )
    states = torch.zeros((2, channels, weight.shape[1] - 1), device='cuda', dtype=x.dtype)
    states[1].copy_(initial[0])
    return function(
        x.t(),
        weight,
        bias,
        states,
        cu,
        cache_indices=torch.ones(1, device='cuda', dtype=torch.int32),
        has_initial_state=torch.ones(1, device='cuda', dtype=torch.bool),
        activation='silu',
        metadata=None,
    ).t()


def scan_call(engine, values, initial=None, state_dtype=torch.float32, raw=False):
    x, dt, A, B, C, D, bias = (values[k] for k in ('x', 'dt', 'A', 'B', 'C', 'D', 'dt_bias'))
    n = x.shape[0]
    cu = torch.tensor([0, n], dtype=torch.int32, device='cuda')
    chunks = torch.tensor(list(range(0, n, 128)) + [n], dtype=torch.int32, device='cuda')
    last = torch.tensor([len(chunks) - 2], dtype=torch.int32, device='cuda')
    seq = torch.zeros(len(chunks) - 1, dtype=torch.int32, device='cuda')
    if initial is None:
        initial = torch.zeros(
            (1, x.shape[1], x.shape[2], B.shape[2]), dtype=state_dtype, device='cuda'
        )
    out = torch.zeros_like(x)
    kwargs = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=128,
        cu_chunk_seqlens=chunks,
        last_chunk_indices=last,
        seq_idx=seq,
        out=out,
        D=D,
        z=None,
        dt_bias=bias,
        initial_states=initial,
        dt_softplus=True,
        dt_limit=(0.0, float('inf')),
        state_dtype=state_dtype,
    )
    if engine == 'minf':
        kwargs['return_raw_states'] = raw
    else:
        kwargs.update(cu_seqlens=cu, return_intermediate_states=raw)
    states = functions(engine)[1](**kwargs)
    if raw and engine == 'minf':
        states = states[1]
    return out, states


def compare_bits(reference, actual):
    """Report exact storage equality, including signed zero, without tolerance."""
    assert reference.shape == actual.shape
    assert reference.dtype == actual.dtype
    assert torch.isfinite(reference).all() and torch.isfinite(actual).all()
    unequal = reference.contiguous().view(torch.uint8) != actual.contiguous().view(torch.uint8)
    return dict(
        elements=reference.numel(),
        different_values=int((reference != actual).sum()),
        different_bytes=int(unequal.sum()),
        exact=not bool(unequal.any()),
    )


@contextlib.contextmanager
def diagnostic_configs(configs):
    """Temporarily restore recorded/debug configs without contaminating native tuning."""
    saved = []
    for entry in configs:
        kernel = getattr(importlib.import_module(entry['module']), entry['attribute'])
        saved.append(
            (kernel, kernel.configs, kernel.cache.copy(), getattr(kernel, 'best_config', None))
        )
    try:
        pin(configs)
        yield
    finally:
        for kernel, configs, cache, best in saved:
            kernel.configs = configs
            kernel.cache.clear()
            kernel.cache.update(cache)
            if best is not None:
                kernel.best_config = best
            elif hasattr(kernel, 'best_config'):
                del kernel.best_config


def config_dict(config):
    """Serialize the numerical launch parameters."""
    return dict(
        kwargs=config.kwargs,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        num_ctas=config.num_ctas,
        maxnreg=config.maxnreg,
    )


def scan_configs(engine):
    """Record the configurations actually selected by native autotuning."""
    result = []
    for name in ['ssd_chunk_state', 'ssd_chunk_scan', 'ssd_state_passing', 'ssd_bmm']:
        module = importlib.import_module(PREFIX[engine] + '.' + name)
        for attr, kernel in vars(module).items():
            if attr.startswith('_') and getattr(kernel, 'best_config', None) is not None:
                result.append(
                    dict(module=module.__name__, attribute=attr, **config_dict(kernel.best_config))
                )
    return result


def main():
    """Compare native tuning; keep historical and matched-config controls separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minf', type=Path, required=True)
    parser.add_argument('--vllm', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument(
        '--diagnostic-pin-cumsum',
        action='store_true',
        help='Also compare after temporarily copying vLLM cumsum choice; not a native-parity pass',
    )
    args = parser.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(rank)
    out = args.out_dir / f'rank{rank}'
    out.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version('vllm')
    assert version == '0.25.1', version
    result = dict(
        rank=rank,
        vllm=version,
        torch=torch.__version__,
        triton=triton.__version__,
        gpu=torch.cuda.get_device_name(),
        job_id=os.environ.get('SLURM_JOB_ID'),
        sources={},
        autotune_policy={},
        recorded_vllm_replay=[],
        conv=[],
        scan=[],
        diagnostic_cumsum=[],
        complete=False,
    )
    for engine in PREFIX:
        for name in [
            'causal_conv1d_varlen' if engine == 'minf' else 'causal_conv1d',
            'ssd_combined',
            'ssd_chunk_state',
            'ssd_state_passing',
            'ssd_bmm',
            'ssd_chunk_scan',
        ]:
            module = importlib.import_module(PREFIX[engine] + '.' + name)
            path = Path(inspect.getsourcefile(module))
            result['sources'][module.__name__] = dict(
                path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
    # Same candidates and order, no deterministic-config filter or production
    # pin. Kernel timing remains native; selected winners may still differ.
    for mod, attr in [
        ('ssd_chunk_state', '_chunk_cumsum_fwd_kernel'),
        ('ssd_chunk_state', '_chunk_state_fwd_kernel'),
        ('ssd_chunk_scan', '_chunk_scan_fwd_kernel'),
        ('ssd_state_passing', '_state_passing_fwd_kernel'),
        ('ssd_bmm', '_bmm_chunk_fwd_kernel'),
    ]:
        configs = {
            e: [
                config_dict(x)
                for x in getattr(importlib.import_module(PREFIX[e] + '.' + mod), attr).configs
            ]
            for e in PREFIX
        }
        assert configs['minf'] == configs['vllm'], (mod, attr, configs)
        result['autotune_policy'][attr] = configs['vllm']
    runs = {'minf': args.minf, 'vllm': args.vllm}
    for source, run in runs.items():
        responses = json.loads((run / 'responses.json').read_text())
        assert responses and all(row['exact_output_parity'] for row in responses)
        for row in responses:
            case = row['case']
            directory = run / 'captures' / case / f'rank{rank}'
            for kind in ['conv', 'scan']:
                paths = sorted(directory.glob('*_' + kind + '.input.json'))
                assert paths, (directory, kind)
                for path in paths:
                    meta, data = read(path)
                    expected_meta, expected = read(paired(path), preserve_strides=False)
                    n = meta['real_end'] - meta['real_start']
                    if source == 'vllm':
                        assert (
                            meta['function']['sha256']
                            == result['sources'][meta['function']['module']]['sha256']
                        )
                        # This is only a historical capture fidelity check.
                        # Restore prior native caches/configs before comparisons.
                        with diagnostic_configs(expected_meta['scalars']['autotune_configs']):
                            replay = native(source, kind, meta, data, expected_meta)
                            checks = {}
                            for key, value in replay.items():
                                target = expected[key]
                                if key == 'out':
                                    target, value = target[:n], value[:n]
                                checks[key] = compare_bits(target, value)
                        result['recorded_vllm_replay'].append(
                            dict(case=case, kind=kind, start=meta['real_start'], checks=checks)
                        )
                    record = dict(
                        case=case,
                        inputs_from=source,
                        start=meta['real_start'],
                        end=meta['real_end'],
                    )
                    if kind == 'conv':
                        if source == 'minf':
                            x, initial = data['x'][:n], data.get('initial_states')
                            if initial is not None:
                                initial = initial[:1]
                        else:
                            x = data['x'].t()[:n]
                            initial = data['conv_states_selected'][..., -3:]
                            if not bool(data['has_initial_state'].all()):
                                initial = torch.zeros_like(initial)
                        if initial is None:
                            initial = torch.zeros(
                                (1, x.shape[1], 3), dtype=x.dtype, device=x.device
                            )
                        x = x.to(initial.dtype).contiguous()
                        weight, bias = data['weight'].to(x.dtype), data['bias'].to(x.dtype)
                        m = conv_call('minf', x, weight, bias, initial)
                        v = conv_call('vllm', x, weight, bias, initial)
                        checks = {'output': compare_bits(v, m)}
                    else:
                        values = {
                            k: (data[k][:n] if k in ('x', 'dt', 'B', 'C') else data[k]).contiguous()
                            for k in ('x', 'dt', 'A', 'B', 'C', 'D', 'dt_bias')
                        }
                        initial = data.get('initial_states')
                        if initial is not None:
                            initial = initial[:1]
                        m, ms = scan_call('minf', values, initial, raw=True)
                        v, vs = scan_call('vllm', values, initial, raw=True)
                        checks = {'output': compare_bits(v, m), 'states': compare_bits(vs, ms)}
                        record['native_configs'] = {e: scan_configs(e) for e in PREFIX}
                        if args.diagnostic_pin_cumsum:
                            selected = [
                                dict(c, module=PREFIX['minf'] + '.ssd_chunk_state')
                                for c in record['native_configs']['vllm']
                                if c['attribute'] == '_chunk_cumsum_fwd_kernel'
                            ]
                            assert len(selected) == 1
                            with diagnostic_configs(selected):
                                dm, ds = scan_call('minf', values, initial, raw=True)
                            result['diagnostic_cumsum'].append(
                                dict(
                                    **record,
                                    checks={
                                        'output': compare_bits(v, dm),
                                        'states': compare_bits(vs, ds),
                                    },
                                    diagnostic_only=True,
                                )
                            )
                    result[kind].append(dict(**record, checks=checks))
                    (out / 'replay.json').write_text(json.dumps(result, indent=2) + '\n')
                    print(
                        kind,
                        rank,
                        case,
                        source,
                        all(x['exact'] for x in checks.values()),
                        flush=True,
                    )
    result['complete'] = True
    result['recorded_replay_exact'] = all(
        c['exact'] for row in result['recorded_vllm_replay'] for c in row['checks'].values()
    )
    result['native_all_exact'] = all(
        c['exact']
        for kind in ('conv', 'scan')
        for row in result[kind]
        for c in row['checks'].values()
    )
    result['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'replay.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['recorded_replay_exact'], 'Historical vLLM replay failed'
    assert result[
        'native_all_exact'
    ], 'Native parity is incomplete; diagnostic pinning does not count'


if __name__ == '__main__':
    main()
