#!/usr/bin/env python3
"""Strict E0 load, production-width branch forward/backward and checkpoint smoke."""
import argparse
import io
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.e18_protocol import E0_SHA256, base, load_checkpoint, write_report
from tools.smoke_e15_model import inputs


def exercise_model(model, device):
    torch.manual_seed(100)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.route_attention.parameters():
        p.requires_grad_(True)
    data = {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs(model.config.hidden_size).items()
            if not key.startswith('gmap_transport_')}
    masks = data['gmap_masks'][:, None, :].expand(-1, 5, -1).clone()
    masks[:, 3] = torch.tensor([False, True, False, True, False], device=device)
    masks[0, 4] = torch.tensor([False, True, True, False, True], device=device)
    data['gmap_route_masks'] = masks
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model.route_attention.full_graph = True
    with torch.no_grad():
        control = model.forward_navigation(**data)
    expected = control['base_global_logits'].log_softmax(-1)
    finite = torch.isfinite(expected)
    mismatch = (expected[finite] - control['global_logits'][finite]).abs().max().item()
    if mismatch > 1e-6:
        raise RuntimeError('Full-graph cloned branch failed initial E0 identity')
    branch = model.route_attention
    model.route_attention = None
    with torch.no_grad():
        disabled = model.forward_navigation(**data)['global_logits']
    model.route_attention = branch
    if not torch.equal(disabled, control['base_global_logits']):
        raise RuntimeError('Disabled E18 changed E0 logits')
    trainable = sum(p.numel() for p in branch.parameters())
    for full_graph in (True, False):
        branch.full_graph = full_graph
        model.zero_grad(set_to_none=True)
        output = model.forward_navigation(**data)
        probs = output['global_logits'].exp()
        if (not torch.isfinite(probs).all() or
                not torch.allclose(probs.sum(-1), torch.ones(2, device=device)) or
                not torch.allclose(probs[:, 0], expected.exp()[:, 0], atol=1e-7) or
                not torch.equal(output['factorized_greedy_actions'] == 0, disabled.argmax(-1) == 0)):
            raise RuntimeError('E18 violated finite distribution or frozen STOP contract')
        (-output['global_logits'][:, 3].mean()).backward()
        gradient = 0.
        for name, p in model.named_parameters():
            if p.grad is not None:
                if not name.startswith('route_attention.') or not torch.isfinite(p.grad).all():
                    raise RuntimeError('Invalid gradient: ' + name)
                gradient += p.grad.abs().sum().item()
        if gradient == 0:
            raise RuntimeError('No navigation gradient reached E18 branch')
    # Round trip only the added branch; a full E0 checkpoint was loaded above.
    stream = io.BytesIO()
    torch.save(branch.state_dict(), stream)
    stream.seek(0)
    branch.load_state_dict(load_checkpoint(stream), strict=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    return dict(trainable_parameters=trainable,
                total_model_parameters=sum(p.numel() for p in model.parameters()),
                full_graph_initial_max_error=mismatch, frozen_gradient=0,
                stop_contract=True, both_arms_finite_backward=True,
                seconds=time.monotonic() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models
    if base.sha256(args.baseline) != E0_SHA256:
        raise RuntimeError('Wrong strict E0 checkpoint')
    config = get_config('run_r2r/iter_train.yaml', [
        'MODEL.route_attention', 'True', 'MODEL.factorized_landmark_size', '0',
        'MODEL.pretrained_path', args.pretrained,
    ])
    model = get_vlnbert_models(config.MODEL)
    checkpoint = load_checkpoint(args.baseline)
    weights = {key.replace('net.module.', 'net.').split('net.vln_bert.', 1)[1]: value
               for key, value in checkpoint['state_dict'].items()
               if key.replace('net.module.', 'net.').startswith('net.vln_bert.')}
    missing = model.load_state_dict(weights, strict=False)
    expected = {'route_attention.' + key for key in model.route_attention.state_dict()}
    if set(missing.missing_keys) != expected or missing.unexpected_keys:
        raise RuntimeError('Strict E0 model load mismatch: ' + str(missing))
    model.route_attention.copy_from_e0(model)
    del checkpoint, weights
    report = exercise_model(model, args.device)
    report.update(experiment='E18-ERA', baseline_sha256=E0_SHA256,
                  device=args.device, model_config=str(model.config))
    write_report(args.output, report)
    print('E18_MODEL_SMOKE_PASSED')


if __name__ == '__main__':
    main()
