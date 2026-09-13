"""Gráficos de comparação dos 3 modelos: curvas de loss por época e
comparação de mAP entre YOLOv8, Faster R-CNN e SSD.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_csv_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def plot_loss_curves(model_name: str, logs_dir: str | Path = "logs",
                      checkpoints_dir: str | Path = "checkpoints",
                      csv_path: str | Path | None = None,
                      output_dir: str | Path = "logs") -> Path:
    """Lê o CSV de loss de `model_name` e plota train loss vs val loss por
    época, salvando como PNG em `output_dir`.

    Para "fasterrcnn"/"ssd", lê por padrão `{logs_dir}/{model_name}_loss.csv`
    (gerado por `train.train_torchvision_model`, colunas: epoch, train_loss,
    val_loss). Para "yolo", lê por padrão `{checkpoints_dir}/yolo/results.csv`
    (gerado automaticamente pelo ultralytics) e soma as 3 componentes de loss
    (box/cls/dfl) de treino e de validação, pra ter uma curva de "loss total"
    comparável às dos outros dois modelos. Use `csv_path` pra apontar pra
    outro arquivo (ex: se o treino do YOLO usou um `--name` diferente).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if csv_path is not None:
        csv_path = Path(csv_path)
    elif model_name == "yolo":
        csv_path = Path(checkpoints_dir) / "yolo" / "results.csv"
    else:
        csv_path = Path(logs_dir) / f"{model_name}_loss.csv"

    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV vazio ou não encontrado: {csv_path}")

    epochs = [int(row["epoch"]) for row in rows]

    if model_name == "yolo":
        train_loss = [
            float(row["train/box_loss"]) + float(row["train/cls_loss"]) + float(row["train/dfl_loss"])
            for row in rows
        ]
        val_loss = [
            float(row["val/box_loss"]) + float(row["val/cls_loss"]) + float(row["val/dfl_loss"])
            for row in rows
        ]
    else:
        train_loss = [float(row["train_loss"]) for row in rows]
        val_loss = [float(row["val_loss"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", label="train loss")
    ax.plot(epochs, val_loss, marker="o", label="val loss")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title(f"Curva de loss — {model_name}")
    ax.legend()
    fig.tight_layout()

    fig_path = output_dir / f"{model_name}_loss_curve.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Gráfico salvo em {fig_path}")
    return fig_path


def plot_map_comparison(results_dict: dict, output_dir: str | Path = "logs") -> Path:
    """Gera um gráfico de barras comparando mAP@0.5 e mAP@0.5:0.95 dos
    modelos em `results_dict` (formato de `eval.evaluate_all_models`:
    `{model_name: {"mAP@0.5": ..., "mAP@0.5:0.95": ...}}`), lado a lado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(results_dict.keys())
    map50 = [results_dict[name]["mAP@0.5"] for name in model_names]
    map50_95 = [results_dict[name]["mAP@0.5:0.95"] for name in model_names]

    x = range(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], map50, width, label="mAP@0.5")
    ax.bar([i + width / 2 for i in x], map50_95, width, label="mAP@0.5:0.95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1)
    ax.set_title("Comparação de mAP entre modelos")
    ax.legend()
    fig.tight_layout()

    fig_path = output_dir / "map_comparison.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Gráfico salvo em {fig_path}")
    return fig_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gera os gráficos de loss e de comparação de mAP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    loss_parser = subparsers.add_parser("loss", help="Plota a curva de loss de um modelo")
    loss_parser.add_argument("model", choices=["fasterrcnn", "ssd", "yolo"])
    loss_parser.add_argument("--logs-dir", default="logs")
    loss_parser.add_argument("--checkpoints-dir", default="checkpoints")

    map_parser = subparsers.add_parser("map", help="Plota a comparação de mAP entre os 3 modelos")
    map_parser.add_argument("--metrics-path", default="logs/metrics.json",
                             help="JSON salvo por eval.evaluate_all_models()")
    map_parser.add_argument("--output-dir", default="logs")

    args = parser.parse_args()

    if args.command == "loss":
        plot_loss_curves(args.model, logs_dir=args.logs_dir, checkpoints_dir=args.checkpoints_dir)
    else:
        with open(args.metrics_path) as f:
            results = json.load(f)
        plot_map_comparison(results, output_dir=args.output_dir)
