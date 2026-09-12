"""Behavioral tests for controller-aligned E18 representations."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_e15_integration import model, navigation_inputs


@pytest.fixture
def era(model):
    config = deepcopy(model.config)
    config.factorized_landmark_size = 0
    config.route_attention = True
    config.route_attention_full_graph = False
    result = type(model)(config).eval()
    result.route_attention.copy_from_e0(result)
    for p in result.parameters():
        p.requires_grad_(False)
    for p in result.route_attention.parameters():
        p.requires_grad_(True)
    return result


def inputs():
    data = {k: v for k, v in navigation_inputs().items()
            if not k.startswith('gmap_transport_')}
    mask = data['gmap_masks'][:, None, :].expand(-1, 5, -1).clone()
    mask[:, 3] = torch.tensor([False, True, False, True, False])
    mask[0, 4] = torch.tensor([False, True, True, False, True])
    data['gmap_route_masks'] = mask
    return data


def test_controller_routes_use_actual_front_and_variable_padding(era):
    from vlnce_baselines.route_attention import batch_executable_route_masks
    graph = SimpleNamespace(
        ghost_pos={'g0': None, 'g1': None},
        front_to_ghost_dist=lambda g: (1.0, 'b' if g == 'g0' else 'a'),
        shortest_path={'a': {'a': ['a'], 'b': ['a', 'b']}},
    )
    other = SimpleNamespace(ghost_pos={}, shortest_path={})
    mask = batch_executable_route_masks(
        [graph, other], ['a', 'x'], [[None, 'a', 'b', 'g0', 'g1'], [None, 'x']],
        5, torch.device('cpu'))
    assert mask.shape == (2, 5, 5)
    assert mask[0, 3].tolist() == [False, True, True, True, False]
    assert mask[0, 4].tolist() == [False, True, False, False, True]
    assert not mask[1, :, 2:].any()
    graph.shortest_path['a']['b'] = ['a', 'missing', 'b']
    with pytest.raises(ValueError, match='route'):
        batch_executable_route_masks([graph], ['a'], [[None, 'a', 'b', 'g0', 'g1']], 5, 'cpu')


def test_full_graph_clone_recovers_e0_and_route_changes_representation(era):
    data = inputs()
    era.route_attention.full_graph = True
    control = era.forward_navigation(**data)
    expected = control['base_global_logits'].log_softmax(-1)
    finite = torch.isfinite(expected)
    assert torch.allclose(control['global_logits'][finite], expected[finite], atol=1e-6)
    era.route_attention.full_graph = False
    route = era.forward_navigation(**data)
    assert not torch.allclose(route['global_logits'][finite], expected[finite], atol=1e-7)
    assert torch.allclose(route['global_logits'].exp()[:, 0], expected.exp()[:, 0], atol=1e-7)
    assert torch.equal(route['factorized_greedy_actions'] == 0,
                       route['base_global_logits'].argmax(-1) == 0)
    assert torch.allclose(route['global_logits'].exp().sum(-1), torch.ones(2))


def test_finite_backward_only_branch_and_checkpoint_roundtrip(era):
    data = inputs()
    output = era.forward_navigation(**data)
    (-output['global_logits'][:, 3].mean()).backward()
    gradients = []
    for name, p in era.named_parameters():
        if name.startswith('route_attention.'):
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
                gradients.append(p.grad.abs().sum())
        else:
            assert p.grad is None, name
    assert sum(gradients) > 0
    clone = deepcopy(era)
    clone.load_state_dict(era.state_dict(), strict=True)
    assert torch.equal(clone.forward_navigation(**data)['global_logits'], output['global_logits'])
    branch = era.route_attention
    era.route_attention = None
    disabled = era.forward_navigation(**data)
    assert torch.equal(disabled['global_logits'], output['base_global_logits'])
    era.route_attention = branch


def test_empty_frontiers_and_invalid_masks(era):
    data = inputs()
    data['gmap_visited_masks'][:, 1:] = True
    out = era.forward_navigation(**data)
    assert torch.equal(out['global_logits'].exp()[:, 0], torch.ones(2))
    assert not torch.isnan(out['global_logits']).any()
    data = inputs()
    for key in ('txt_masks', 'gmap_masks', 'gmap_route_masks'):
        broken = dict(data)
        broken[key] = torch.zeros_like(data[key])
        with pytest.raises(ValueError, match='mask'):
            era.forward_navigation(**broken)
    broken = dict(data, gmap_route_masks=torch.ones(2, 4, 5, dtype=torch.bool))
    with pytest.raises(ValueError, match='shape'):
        era.forward_navigation(**broken)


def test_e15_and_e18_cannot_be_combined(model):
    config = deepcopy(model.config)
    config.route_attention = True
    with pytest.raises(ValueError, match='E18'):
        type(model)(config)


def test_training_recipe_rejects_surface_variants(era):
    from vlnce_baselines.route_attention import validate_training_config
    flags = dict(route_attention_only=True, sample_num=8, update_epochs=1,
                 batch_size=1, lr=2e-5, min_lr_ratio=.25, warmup_iters=0,
                 grpo_beta=.04, max_grad_norm=2., load_from_ckpt=True,
                 back_algo='control', is_requeue=False, enable_amp=False,
                 enable_all_dropouts=False, dropout_in_sampling=False,
                 waypoint_aug=False)
    config = SimpleNamespace(
        GRPO=SimpleNamespace(**flags), MODEL=SimpleNamespace(route_attention=True, task_type='r2r'),
        GPU_NUMBERS=1, NUM_ENVIRONMENTS=1,
        TASK_CONFIG=SimpleNamespace(DATASET=SimpleNamespace(SPLIT='train', SUFFIX='_10')))
    validate_training_config(config)
    for field, value in [('lr', 1e-4), ('factorized_landmark_only', True), ('is_requeue', True)]:
        bad = deepcopy(config)
        setattr(bad.GRPO, field, value)
        with pytest.raises(ValueError, match='E18'):
            validate_training_config(bad)


def test_production_policy_passes_route_masks_and_smoke_runs(era):
    import ast
    from pathlib import Path
    from tools.smoke_e18_model import exercise_model
    source = (Path(__file__).resolve().parents[1] / 'vlnce_baselines/models/R1Policy.py').read_text()
    tree = ast.parse(source)
    forward = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == 'forward'
                   and any(arg.arg == 'mode' for arg in node.args.args))
    namespace = {}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), '<production forward>', 'exec'), namespace)
    data = inputs()
    data['gmap_vp_ids'] = data.pop('gmap_vpids')
    output = namespace['forward'](SimpleNamespace(vln_bert=era), mode='navigation', **data)
    assert output['global_logits'].shape == (2, 5)
    report = exercise_model(era, 'cpu')
    assert report['both_arms_finite_backward']
    assert report['full_graph_initial_max_error'] < 1e-6
