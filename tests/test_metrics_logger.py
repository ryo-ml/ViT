import csv

from src.utils.metrics_logger import MetricsLogger

def test_metrics_logger(tmp_path) -> None:
    metrics_logger = MetricsLogger(tmp_path, use_wandb=False)

    metrics = {
        'step': 1,
        'train/loss': 0.1,
        'train/acc': 0.9,
    }
    metrics_logger.log(1, metrics)

    metrics = {
        'step': 2,
        'val/loss': 0.2,
        'val/acc': 0.8,
    }
    metrics_logger.log(2, metrics)

    with open(tmp_path / 'metrics.csv', 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    assert fieldnames == ['step', 'train/loss', 'train/acc', 'val/loss', 'val/acc']

    assert len(rows) == 2

    assert rows[0]['step'] == '1'
    assert rows[0]['train/loss'] == '0.1'
    assert rows[0]['train/acc'] == '0.9'
    assert rows[0]['val/loss'] == ''
    assert rows[0]['val/acc'] == ''

    assert rows[1]['step'] == '2'
    assert rows[0]['train/loss'] == '0.1'
    assert rows[1]['train/loss'] == ''
    assert rows[1]['train/acc'] == ''
    assert rows[1]['val/loss'] == '0.2'
    assert rows[1]['val/acc'] == '0.8'

    metrics_logger.close()
