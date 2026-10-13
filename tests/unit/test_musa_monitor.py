import json
from types import SimpleNamespace
import pytest
import torch
from worldscape_policy.training.musa_monitor import MusaStepMonitor


def test_update_evidence_and_nonfinite_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, 'musa', SimpleNamespace(
        reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 123), raising=False)
    policy = torch.nn.Linear(3, 2)
    trainer = SimpleNamespace(policy=policy, runtime=SimpleNamespace(rank=0), step=1)
    monitor = MusaStepMonitor(tmp_path)
    monitor.on_train_start(trainer)
    with torch.no_grad():
        policy.weight.add_(0.25)
    monitor.on_after_step(trainer, None, {'loss': 1.0})
    row = json.loads((tmp_path/'rank-0-steps.jsonl').read_text())
    assert row['sampled_parameter_max_delta']['weight'] == pytest.approx(0.25)
    assert row['sampled_parameter_max_delta']['bias'] == 0
    with pytest.raises(FloatingPointError):
        monitor.on_after_step(trainer, None, {'loss': float('nan')})
