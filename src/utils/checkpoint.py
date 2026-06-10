from pathlib import Path
import math
import torch

class Checkpointer:
    def __init__(
        self,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.best_metric = -math.inf

    def save(
        self,
        model: torch.nn.Module,
        step: int,
        metrics: dict,
        filename: str,
    ) -> None:
        d = {
            'model_state_dict': model.state_dict(),
            'step': step,
            'metrics': metrics,
        }
        torch.save(
            d,
            self.output_dir / filename,
        )

    @staticmethod
    def load(
        path: Path,
        model: torch.nn.Module,
        map_location: torch.device,
    ) -> None:
        d = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(d['model_state_dict'])

    def is_best(self, metric: float) -> bool:
        is_best =  metric > self.best_metric
        if is_best:
            self.best_metric = metric
        return is_best
