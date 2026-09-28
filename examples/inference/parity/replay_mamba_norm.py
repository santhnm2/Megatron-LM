# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare the branch's norm to the installed vLLM method, eager and compiled."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

import torch
from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated

from megatron.core.ssm.mamba_mixer import ExtendedRMSNorm
from megatron.core.ssm.ops.vllm_grouped_rmsnorm import (
    compiled_grouped_gated_rmsnorm,
    grouped_gated_rmsnorm,
)


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return dict(
        elements=a.numel(),
        different_bytes=int(
            (a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum()
        ),
        different_values=int((a != b).sum()),
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ['minf', 'vllm', 'out-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    out = args.out_dir / f'rank{rank}'
    out.mkdir(parents=True, exist_ok=True)

    # Use the actual installed method unbound, with the exact noncollective
    # group attributes, so the reference remains independent of branch code.
    class VNorm(torch.nn.Module):
        forward = Mixer2RMSNormGated.forward_native

        def __init__(self, w):
            super().__init__()
            self.weight = torch.nn.Parameter(w, requires_grad=False)
            self.n_groups, self.tp_size, self.tp_rank = 8, 4, rank
            self.group_size, self.per_rank_hidden_size = 512, 1024
            self.use_rms_norm, self.variance_epsilon = True, 1e-5
            self.eps = 1e-5

    vr = args.vllm
    mr = args.minf
    (wp,) = (mr / f'captures/block_weights/rank{rank}').glob('*_decoder.layers.0.parameters.pt')
    w = torch.load(wp, weights_only=True, map_location='cpu')['mixer.norm.weight.'].cuda()
    v = VNorm(w).eval()
    vc = torch.compile(v, fullgraph=True, dynamic=True)
    m = ExtendedRMSNorm(
        1024, eps=1e-5, group_size=512, norm_before_gate=False, device='cuda', dtype=torch.bfloat16
    ).eval()
    with torch.no_grad():
        m.weight.copy_(w)
    reference_file = Path(inspect.getsourcefile(Mixer2RMSNormGated))
    result = dict(
        rows=[],
        source=str(reference_file),
        reference_sha256=hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        versions={name: importlib.metadata.version(name) for name in ['torch', 'triton', 'vllm']},
        rank=rank,
        complete=False,
    )
    for engine, run, prefix in [('minf', mr, 'decoder'), ('vllm', vr, 'model')]:
        for p in sorted(
            (run / 'captures').glob(f'*/rank{rank}/*_{prefix}.layers.0.mixer.norm.input.pt')
        ):
            tensors = torch.load(p, map_location='cpu', weights_only=True)
            meta = json.loads(p.with_suffix('.json').read_text())['tensors']
            restored = []
            for key in ['args.0.', 'args.1.']:
                a = tensors[key]
                g = torch.empty_strided(a.shape, meta[key]['stride'], dtype=a.dtype, device='cuda')
                g.copy_(a)
                restored.append(g)
            x, z = restored
            xx, zz = [a.reshape(-1, 1024) for a in restored]
            with torch.no_grad():
                re = v(xx, zz)
                rc = vc(xx, zz)
                me = grouped_gated_rmsnorm(v, xx, zz)
                mc = m(x, z).reshape(-1, 1024)
            result['rows'].append(
                dict(
                    source=engine,
                    case=p.parent.parent.name,
                    capture=p.name,
                    rows=xx.shape[0],
                    strides=[list(a.stride()) for a in restored],
                    eager=bits(re, me),
                    compiled=bits(rc, mc),
                    reference_eager_vs_compiled=bits(re, rc),
                )
            )
            (out / 'norm.json').write_text(json.dumps(result, indent=2) + '\n')
    result['complete'] = True
    result['all_exact'] = all(
        r[k]['different_bytes'] == 0 for r in result['rows'] for k in ['eager', 'compiled']
    )
    (out / 'norm.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['all_exact'], 'Byte mismatch; see norm.json'


if __name__ == '__main__':
    main()
