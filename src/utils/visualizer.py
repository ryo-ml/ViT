from pathlib import Path
import matplotlib.pyplot as plt

def plot_metrics(
    x: list,
    y: list,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x, y)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    fig.savefig(path, dpi=400)

    plt.close(fig)
