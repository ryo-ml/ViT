from pathlib import Path
import os
from dotenv import load_dotenv
import wandb
import csv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class MetricsLogger:
    def __init__(
        self,
        output_dir: Path,
        use_wandb: bool,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / 'metrics.csv'
        self._fieldnames = []
        self.use_wandb = use_wandb

        if self.output_path.exists():
            with open(self.output_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                self._fieldnames = list(reader.fieldnames or [])

        if use_wandb:
            load_dotenv(BASE_DIR / '.env')
            wandb.login(os.environ['WANDB_API_KEY'])
            self.run = wandb.init()

    def _write_csv(self, step: int, metrics: dict) -> None:
        row = {'step': step, **metrics}
        new_keys = [k for k in row.keys() if k not in self._fieldnames]

        if new_keys:
            self._fieldnames.extend(new_keys)

            existing_rows = []
            if self.output_path.exists():
                with open(self.output_path, 'r', newline='') as f:
                    existing_rows = list(csv.DictReader(f))

            with open(self.output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)

        with open(self.output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)

    def log(self, step: int, metrics: dict) -> None:
        self._write_csv(step, metrics)

        if self.use_wandb:
            self.run.log(metrics, step=step)

    def close(self) -> None:
        if self.use_wandb:
            self.run.finish()
