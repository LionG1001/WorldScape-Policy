"""External measurement callback; does not change the source training loop."""
import json
import time
import torch
from worldscape_policy.training.musa_monitor import MusaStepMonitor


class PerformanceMonitor(MusaStepMonitor):
    def on_train_start(self, trainer):
        super().on_train_start(trainer)
        self.perf_path = self.output_dir / f'rank-{trainer.runtime.rank}-performance.jsonl'
        self.previous_end = time.perf_counter()
        self.loop_start = self.previous_end
        metadata = dict(rank=trainer.runtime.rank, world_size=trainer.runtime.world_size,
                        device=str(trainer.runtime.device),
                        trainable_parameters=sum(p.numel() for p in trainer.policy.parameters() if p.requires_grad),
                        python=__import__('sys').version, torch=torch.__version__)
        (self.output_dir / f'rank-{trainer.runtime.rank}-environment.json').write_text(json.dumps(metadata, indent=2))

    def on_before_step(self, trainer, batch):
        self.step_start = time.perf_counter()
        self.gap = self.step_start - self.previous_end

    def on_after_step(self, trainer, batch, metrics):
        super().on_after_step(trainer, batch, metrics)
        torch.musa.synchronize()
        end = time.perf_counter()
        shapes = {key: list(value.shape) for key, value in batch.observations.items()
                  if isinstance(value, torch.Tensor)}
        row = dict(rank=trainer.runtime.rank, step=trainer.step,
                   wall_unix_s=time.time(), elapsed_loop_s=end-self.loop_start,
                   interval_s=end-self.previous_end, train_s=end-self.step_start,
                   between_steps_s=self.gap, batch_size=int(batch.actions.shape[0]),
                   action_shape=list(batch.actions.shape), observation_shapes=shapes,
                   allocated_bytes=torch.musa.memory_allocated(),
                   reserved_bytes=torch.musa.memory_reserved(),
                   peak_allocated_bytes=torch.musa.max_memory_allocated(),
                   peak_reserved_bytes=torch.musa.max_memory_reserved())
        with self.perf_path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.previous_end = end
