"""Small per-rank evidence log for short MUSA bring-up runs."""
import json
import math
from pathlib import Path

import torch

from worldscape_policy.training.callbacks import NoOpTrainingCallback


class MusaStepMonitor(NoOpTrainingCallback):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def on_train_start(self, trainer):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"rank-{trainer.runtime.rank}-steps.jsonl"
        self.samples = []
        # Sample several trainable tensors, without retaining a model copy.
        for name, parameter in trainer.policy.named_parameters():
            if parameter.requires_grad and len(self.samples) < 8:
                before = parameter.detach().reshape(-1)[:4096].float().clone()
                self.samples.append((name, parameter, before))
        torch.musa.reset_peak_memory_stats()

    def on_after_step(self, trainer, batch, metrics):
        del batch
        values = {key: float(value) for key, value in metrics.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"non-finite training metrics: {values}")
        # Reduce on device and copy eight scalars once, instead of copying eight
        # parameter slices and synchronizing once per slice on every step.
        changes = [
            (parameter.detach().reshape(-1)[:4096].float() - before).abs().max()
            for _, parameter, before in self.samples
        ]
        values_cpu = torch.stack(changes).cpu().tolist() if changes else []
        deltas = {name: value for (name, _, _), value in zip(self.samples, values_cpu)}
        record = dict(rank=trainer.runtime.rank, step=trainer.step, metrics=values,
                      sampled_parameter_max_delta=deltas,
                      peak_allocated_bytes=torch.musa.max_memory_allocated())
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
