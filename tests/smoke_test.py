"""Smoke test: roda 1 época com um subconjunto pequeno de dados (CPU) pros
3 modelos, só pra garantir que data.py, models.py, train.py e eval.py
funcionam sem erro antes de rodar de verdade no Colab.

Não gera métricas úteis (poucas imagens, 1 época) — só valida que o
pipeline inteiro roda sem quebrar. Usa uma pasta temporária própria
(apagada no final), sem tocar em data/, checkpoints/ ou logs/ do projeto.

Uso: python tests/smoke_test.py [--limit N] [--epochs N] [--batch-size N] [--keep]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

import data as data_module
import eval as eval_module
import models
import train


def run_smoke_test(limit: int = 20, epochs: int = 1, batch_size: int = 2, keep: bool = False) -> bool:
    tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_test_"))
    data_dir = tmp_dir / "data"
    checkpoints_dir = tmp_dir / "checkpoints"
    logs_dir = tmp_dir / "logs"

    print(f"Diretório temporário: {tmp_dir}")
    ok = True
    try:
        print(f"\n[1/4] data.py: convertendo {limit} imagens (Fashionpedia -> COCO/YOLO)...")
        counts = data_module.convert_fashionpedia(output_dir=data_dir, limit=limit)
        print(f"      OK: {counts}")

        num_classes = len(data_module.get_category_names())

        print("\n[2/4] models.py: construindo os 3 modelos...")
        fasterrcnn = models.get_fasterrcnn_model(num_classes)
        ssd = models.get_ssd_model(num_classes)
        yolo = models.get_yolo_model()
        print("      OK")

        print(f"\n[3/4] train.py: treinando {epochs} época(s) de cada modelo em CPU...")

        train_ds = data_module.CocoDetectionDataset(
            data_dir / "coco" / "images" / "train", data_dir / "coco" / "annotations" / "instances_train.json"
        )
        val_ds = data_module.CocoDetectionDataset(
            data_dir / "coco" / "images" / "val", data_dir / "coco" / "annotations" / "instances_val.json"
        )

        train.train_torchvision_model(
            fasterrcnn, train_ds, val_ds, model_name="fasterrcnn", device="cpu",
            epochs=epochs, batch_size=batch_size, checkpoints_dir=checkpoints_dir, logs_dir=logs_dir,
        )
        print("      OK: fasterrcnn")

        train.train_torchvision_model(
            ssd, train_ds, val_ds, model_name="ssd", device="cpu",
            epochs=epochs, batch_size=batch_size, checkpoints_dir=checkpoints_dir, logs_dir=logs_dir,
        )
        print("      OK: ssd")

        train.train_yolo_model(
            yolo, data_dir / "yolo" / "data.yaml", epochs=epochs, batch_size=batch_size,
            project=checkpoints_dir, name="yolo",
        )
        print("      OK: yolo")

        print("\n[4/4] eval.py: avaliando os 3 modelos no conjunto de teste...")
        fasterrcnn_ckpt = sorted(checkpoints_dir.glob("fasterrcnn_epoch*.pt"))[-1]
        ssd_ckpt = sorted(checkpoints_dir.glob("ssd_epoch*.pt"))[-1]

        fasterrcnn.load_state_dict(torch.load(fasterrcnn_ckpt, map_location="cpu")["model_state_dict"])
        ssd.load_state_dict(torch.load(ssd_ckpt, map_location="cpu")["model_state_dict"])
        yolo_trained = models.get_yolo_model(str(checkpoints_dir / "yolo" / "weights" / "best.pt"))

        results = eval_module.evaluate_all_models(
            {"fasterrcnn": fasterrcnn, "ssd": ssd, "yolo": yolo_trained},
            data_dir=data_dir, predictions_dir=logs_dir, batch_size=batch_size,
        )
        print(f"      OK: {results}")

    except Exception:
        ok = False
        print("\nFALHOU:")
        traceback.print_exc()
    finally:
        if keep:
            print(f"\n--keep: mantendo {tmp_dir} pra inspeção manual.")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="Quantidade de imagens de treino a baixar")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--keep", action="store_true", help="Não apaga a pasta temporária no final")
    args = parser.parse_args()

    success = run_smoke_test(limit=args.limit, epochs=args.epochs, batch_size=args.batch_size, keep=args.keep)

    if success:
        print("\nSMOKE TEST OK — data.py, models.py, train.py e eval.py rodaram sem erro.")
        print("Agora sim, pode rodar de verdade no Colab.")
        sys.exit(0)
    else:
        print("\nSMOKE TEST FALHOU — corrija antes de gastar tempo/GPU no Colab.")
        sys.exit(1)
