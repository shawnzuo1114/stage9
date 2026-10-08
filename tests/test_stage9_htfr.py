import importlib.util
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

_STAGE9_PATH = pathlib.Path(__file__).parents[1] / "models" / "stage9_htfr_reid.py"
_SPEC = importlib.util.spec_from_file_location("stage9_htfr_reid_test_module", _STAGE9_PATH)
_STAGE9 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STAGE9)
HierarchicalTemporalFrequencyRouter = _STAGE9.HierarchicalTemporalFrequencyRouter
TemporalHighPass = _STAGE9.TemporalHighPass

_EVALUATOR_PATH = pathlib.Path(__file__).parents[1] / "engine" / "evaluator.py"
_EVAL_SPEC = importlib.util.spec_from_file_location("stage9_evaluator_test_module", _EVALUATOR_PATH)
_EVALUATOR = importlib.util.module_from_spec(_EVAL_SPEC)
_EVAL_SPEC.loader.exec_module(_EVALUATOR)


def _module(mode="full"):
    return HierarchicalTemporalFrequencyRouter(
        stage3_channels=8,
        stage4_channels=12,
        token_dim=16,
        query_dim=4,
        temporal_temperature=1.0,
        gate_max=0.20,
        fusion_start_epoch=15,
        fusion_warmup_epoch=10,
        mode=mode,
    )


def _inputs(batch=4, seq_len=12):
    torch.manual_seed(7)
    return (
        torch.rand(batch, seq_len, 8, 5, 3),
        torch.rand(batch, seq_len, 12, 3, 2),
        F.layer_norm(torch.randn(batch, 16), (16,)),
    )


def _grad_norm(module):
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return torch.stack(values).norm().item() if values else 0.0


def test_temporal_high_pass_extreme_lengths():
    hpf = TemporalHighPass()
    for seq_len in (1, 2, 12):
        x = torch.randn(3, seq_len, 5)
        y, stats = hpf(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert all(torch.isfinite(value) for value in stats.values())
    y_one, _ = hpf(torch.randn(2, 1, 4))
    assert torch.equal(y_one, torch.zeros_like(y_one))


def test_initialization_invariants_and_shapes():
    model = _module("full")
    feat3, feat4, z_g = _inputs()
    output = model(feat3, feat4, z_g, epoch=25)
    assert output["z_final"].shape == (4, 16)
    assert output["v_freq"].shape == (4, 16)
    assert output["attn3"].shape == (4, 12)
    assert output["attn4"].shape == (4, 12)
    assert output["stage_weights"].shape == (4, 2)
    assert abs(output["gate"].mean().item() - 0.04) < 0.002
    assert torch.equal(output["delta"], torch.zeros_like(output["delta"]))
    expected = F.layer_norm(z_g, (16,))
    assert torch.allclose(output["z_final"], expected, atol=1e-6, rtol=1e-5)
    assert F.cosine_similarity(z_g, output["z_final"], dim=-1).mean().item() > 0.9999


def test_clean_ablation_modes():
    feat3, feat4, z_g = _inputs()
    baseline = _module("baseline")(feat3, feat4, z_g, epoch=100)
    spatial = _module("spatial")(feat3, feat4, z_g, epoch=100)
    full = _module("full")(feat3, feat4, z_g, epoch=100)
    assert torch.equal(baseline["z_final"], z_g)
    assert spatial["diagnostics"]["htfr_s3_temporal_hf_norm"] == 0.0
    assert spatial["diagnostics"]["htfr_s4_temporal_hf_norm"] == 0.0
    assert full["diagnostics"]["htfr_s3_temporal_hf_norm"] > 0.0
    assert full["diagnostics"]["htfr_s4_temporal_hf_norm"] > 0.0


def test_fusion_schedule_is_zero_then_linear_then_full():
    model = _module("full")
    assert model.fusion_weight(14) == 0.0
    assert model.fusion_weight(15) == 0.0
    assert model.fusion_weight(20) == 0.5
    assert model.fusion_weight(25) == 1.0
    assert _module("baseline").fusion_weight(100) == 0.0


def test_auxiliary_and_fusion_gradient_health():
    model = _module("full")
    classifier = nn.Linear(16, 3, bias=False)
    feat3, feat4, z_g = _inputs()
    labels = torch.tensor([0, 1, 2, 0])

    auxiliary = model(feat3, feat4, z_g, epoch=10)
    aux_loss = F.cross_entropy(classifier(auxiliary["v_freq"]), labels)
    aux_loss.backward()
    assert _grad_norm(model.temporal_router_s3.q_proj) > 0.0
    assert _grad_norm(model.temporal_router_s3.k_proj) > 0.0
    assert _grad_norm(model.temporal_router_s4.q_proj) > 0.0
    assert _grad_norm(model.temporal_router_s4.k_proj) > 0.0
    assert _grad_norm(model.stage_router.router) > 0.0
    # fusion_warm=0: the auxiliary representation intentionally does not train the gate.
    assert _grad_norm(model.residual_gate.gate_net) == 0.0

    model.zero_grad(set_to_none=True)
    classifier.zero_grad(set_to_none=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    target = torch.randn_like(z_g)
    first = model(feat3, feat4, z_g, epoch=25)
    (first["z_final"] * target).sum().backward()
    assert _grad_norm(model.residual_gate.delta_proj) > 0.0
    optimizer.step()  # makes the zero-initialized final delta projection informative

    optimizer.zero_grad(set_to_none=True)
    second = model(feat3, feat4, z_g, epoch=25)
    (second["z_final"] * target).sum().backward()
    assert _grad_norm(model.residual_gate.gate_net) > 0.0
    assert _grad_norm(model.residual_gate.delta_proj) > 0.0
    assert _grad_norm(model.stage_router.router) > 0.0


def test_no_forward_hard_gate_or_residual_clipping():
    source = pathlib.Path("models/stage9_htfr_reid.py").read_text(encoding="utf-8")
    assert "torch.clamp(gate" not in source
    assert "gate.clamp(" not in source
    assert "delta *= " not in source
    assert "if delta_ratio" not in source


def test_single_modality_stage9_evaluation_outputs():
    class DummySingleModalityModel(nn.Module):
        def forward(self, imgs, modal, seq_len):
            assert modal in (1, 2)
            base = imgs.float().view(imgs.size(0), -1)
            z_g = torch.cat([base, base], dim=1)
            v_freq = torch.cat([base.square(), base.square()], dim=1)
            z_final = z_g + 0.01 * v_freq
            x_pool = torch.cat([z_final, z_final], dim=1)
            bn_feat = x_pool - x_pool.mean(dim=1, keepdim=True)
            return {
                "z_g": z_g, "v_freq": v_freq, "z_final": z_final,
                "x_pool": x_pool, "bn_feat": bn_feat,
            }

    imgs = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    loader = [(imgs, torch.tensor([0, 1]), torch.tensor([1, 2]))]
    feature_sets, pids, camids = _EVALUATOR.extract_loader_stage9_feature_sets(
        DummySingleModalityModel(), loader, modality_mode=2, seq_len=1, device="cpu"
    )
    assert set(feature_sets) == {"z_g", "v_freq", "z_final", "x_pool", "bn_feat"}
    assert pids.tolist() == [0, 1]
    assert camids.tolist() == [1, 2]
    for value in feature_sets.values():
        assert value.shape[0] == 2
        assert torch.allclose(torch.from_numpy(value).norm(dim=1), torch.ones(2), atol=1e-6)
