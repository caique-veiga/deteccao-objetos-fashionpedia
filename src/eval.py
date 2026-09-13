"""Avaliação dos 3 modelos: roda inferência no conjunto de teste, exporta as
predições no formato COCO "results" (lista de detecções) e calcula
mAP@0.5 e mAP@0.5:0.95 com `pycocotools`.

Convenção de labels: Faster R-CNN e SSD (torchvision) preveem labels 1..N
(0 é a classe de background reservada pelo torchvision — ver o comentário
em `data.CocoDetectionDataset`). Ao exportar pro formato COCO, subtraímos 1
pra voltar aos `category_id` 0..N-1 usados no `instances_test.json` gerado
por `src/data.py`. O YOLO já prevê classes 0-indexadas direto (mesma ordem
de categorias do `data.yaml`), sem precisar desse ajuste.

Usamos sempre `data/coco/...` como referência (imagens e ground truth),
mesmo para avaliar o YOLO: as imagens em `data/coco/images/test` e
`data/yolo/images/test` são as mesmas (só duplicadas em formatos
diferentes por `src/data.py`), então não há necessidade de ground truth
separado por modelo — todos avaliam contra o mesmo `instances_test.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from data import CocoDetectionDataset, detection_collate_fn


@torch.no_grad()
def predict_torchvision_model(model, test_dataset: CocoDetectionDataset, device: str | None = None,
                               batch_size: int = 4, score_threshold: float = 0.0) -> list[dict]:
    """Roda inferência de um modelo torchvision (Faster R-CNN ou SSD) em
    `test_dataset` e retorna as predições no formato COCO results:
    lista de {"image_id", "category_id", "bbox": [x,y,w,h], "score"}.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate_fn)

    results = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            boxes = output["boxes"].cpu()
            labels = output["labels"].cpu()
            scores = output["scores"].cpu()

            for box, label, score in zip(boxes, labels, scores):
                if score.item() < score_threshold:
                    continue
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id": image_id,
                    "category_id": int(label.item()) - 1,  # desfaz o +1 do background
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score.item()),
                })
    return results


def predict_yolo_model(yolo_model, images_dir: str | Path, ann_path: str | Path,
                        imgsz: int = 640, score_threshold: float = 0.0,
                        device: str | None = None, batch_size: int = 4) -> list[dict]:
    """Roda inferência do YOLO nas imagens listadas em `ann_path` (COCO json,
    usado só pra saber quais image_id/file_name existem) e retorna as
    predições no formato COCO results.

    Usa `stream=True` + `batch=batch_size`: o ultralytics processa a lista de
    imagens em lotes internamente (mesmo espírito do DataLoader usado por
    `predict_torchvision_model`), em vez de 1 chamada `.predict()` por imagem.
    """
    images_dir = Path(images_dir)
    with open(ann_path) as f:
        coco = json.load(f)

    image_paths = [str(images_dir / img["file_name"]) for img in coco["images"]]
    image_ids = [img["id"] for img in coco["images"]]

    predictions_stream = yolo_model.predict(
        source=image_paths, imgsz=imgsz, batch=batch_size, device=device, stream=True, verbose=False,
    )

    results = []
    for image_id, prediction in zip(image_ids, predictions_stream):
        boxes = prediction.boxes.xyxy.cpu()
        classes = prediction.boxes.cls.cpu()
        scores = prediction.boxes.conf.cpu()

        for box, cls, score in zip(boxes, classes, scores):
            if score.item() < score_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "image_id": image_id,
                "category_id": int(cls.item()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score.item()),
            })
    return results


def compute_coco_metrics(ann_path: str | Path, predictions: list[dict]) -> dict:
    """Calcula mAP@0.5 e mAP@0.5:0.95 com `pycocotools`, comparando
    `predictions` (formato COCO results) contra o ground truth em `ann_path`
    (`instances_test.json`, gerado por `src/data.py`).
    """
    coco_gt = COCO(str(ann_path))

    if not predictions:
        return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0}

    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # coco_eval.stats segue a ordem padrão do pycocotools: [0]=mAP@0.5:0.95
    # (IoU 0.50:0.95), [1]=mAP@0.5 (IoU 0.50).
    return {
        "mAP@0.5:0.95": float(coco_eval.stats[0]),
        "mAP@0.5": float(coco_eval.stats[1]),
    }


def evaluate_model(model_name: str, model, data_dir: str | Path = "data",
                    predictions_dir: str | Path = "logs", device: str | None = None,
                    batch_size: int = 4) -> dict:
    """Avalia um modelo (`model_name` em {"fasterrcnn", "ssd", "yolo"}) no
    conjunto de teste, salva as predições em
    `{predictions_dir}/{model_name}_predictions.json` e retorna um dict com
    mAP@0.5 e mAP@0.5:0.95.
    """
    data_dir = Path(data_dir)
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    images_dir = data_dir / "coco" / "images" / "test"
    ann_path = data_dir / "coco" / "annotations" / "instances_test.json"

    if model_name == "yolo":
        predictions = predict_yolo_model(model, images_dir, ann_path, device=device, batch_size=batch_size)
    else:
        test_dataset = CocoDetectionDataset(images_dir, ann_path)
        predictions = predict_torchvision_model(model, test_dataset, device=device, batch_size=batch_size)

    predictions_path = predictions_dir / f"{model_name}_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(predictions, f)

    metrics = compute_coco_metrics(ann_path, predictions)
    print(f"[{model_name}] mAP@0.5={metrics['mAP@0.5']:.4f} mAP@0.5:0.95={metrics['mAP@0.5:0.95']:.4f} "
          f"({len(predictions)} predições em {predictions_path})")
    return metrics


def evaluate_all_models(models_dict: dict, data_dir: str | Path = "data",
                         predictions_dir: str | Path = "logs", **kwargs) -> dict:
    """Avalia vários modelos de uma vez. `models_dict` é `{model_name: model}`.
    Retorna `{model_name: {"mAP@0.5": ..., "mAP@0.5:0.95": ...}}` e também
    salva esse dict em `{predictions_dir}/metrics.json`, pra `src/plotting.py`
    conseguir gerar o gráfico comparativo depois sem precisar rodar a
    avaliação de novo na mesma sessão.
    """
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    results = {
        name: evaluate_model(name, model, data_dir=data_dir, predictions_dir=predictions_dir, **kwargs)
        for name, model in models_dict.items()
    }

    metrics_path = predictions_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Métricas de todos os modelos salvas em {metrics_path}")

    return results


if __name__ == "__main__":
    import argparse

    import models

    parser = argparse.ArgumentParser(description="Avalia um dos 3 modelos no conjunto de teste (mAP via pycocotools).")
    parser.add_argument("model", choices=["fasterrcnn", "ssd", "yolo"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", required=True,
                         help="Checkpoint a avaliar: .pt do train.py (fasterrcnn/ssd) ou best.pt/last.pt do YOLO")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.model == "yolo":
        model = models.get_yolo_model(args.checkpoint)
    else:
        import data as data_module

        num_classes = len(data_module.get_category_names())
        model = models.get_fasterrcnn_model(num_classes) if args.model == "fasterrcnn" else models.get_ssd_model(num_classes)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(args.model, model, data_dir=args.data_dir, batch_size=args.batch_size)
