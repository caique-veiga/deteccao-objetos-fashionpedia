"""Download e conversão do dataset Fashionpedia (Hugging Face) para os formatos
COCO (json) e YOLO (txt), além de uma função de EDA (distribuição de classes e
tamanho médio das bounding boxes).

Fonte: https://huggingface.co/datasets/detection-datasets/fashionpedia
- O bbox no dataset original vem em Pascal VOC: [x_min, y_min, x_max, y_max],
  em pixels absolutos. COCO usa [x_min, y_min, width, height]; YOLO usa
  [x_center, y_center, width, height] normalizado entre 0 e 1.
- O dataset só tem os splits "train" e "val" no Hugging Face (sem "test"
  rotulado). Por isso dividimos "train" em train/val para uso durante o treino,
  e usamos o "val" original do HF como nosso "test" (nunca visto durante
  o treino/tuning).
- Em modo `limit` (smoke test), NÃO usamos `datasets` em streaming: para este
  dataset o streaming tenta ler um bloco grande do parquet remoto de uma vez
  (um GET HTTP cujo corpo inteiro é bufferizado em RAM antes de retornar),
  o que estoura memória mesmo pedindo poucos exemplos. Em vez disso, baixamos
  1 shard parquet pequeno (cacheado por `huggingface_hub`) e lemos poucas
  linhas dele localmente com `pyarrow`, o que é seguro em memória.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_dataset, load_dataset_builder
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
import os
import torch
import torchvision.transforms.functional as TF

load_dotenv()

HF_DATASET_NAME = "detection-datasets/fashionpedia"
HF_TOKEN = os.getenv("HF_TOKEN") or None
_PARQUET_SHARD = {"train": "default/train/0000.parquet", "val": "default/val/0000.parquet"}


def get_category_names() -> list[str]:
    """Retorna os nomes das 46 categorias do Fashionpedia, na ordem dos ids.

    Usa `load_dataset_builder`, que só baixa metadados (não as imagens).
    """
    builder = load_dataset_builder(HF_DATASET_NAME, token=HF_TOKEN)
    return builder.info.features["objects"]["category"].feature.names


def _load_split_full(hf_split: str) -> Dataset:
    return load_dataset(HF_DATASET_NAME, split=hf_split, token=HF_TOKEN)


def _load_split_limited(hf_split: str, limit: int, seed: int) -> list[dict]:
    """Baixa só 1 shard parquet pequeno (~85MB o de val, ~480MB o de treino) e lê
    `limit` linhas dele localmente com pyarrow. Usado no smoke test e em
    execuções rápidas de depuração — ver nota no docstring do módulo sobre por
    que não usamos o modo streaming do `datasets` aqui.
    """
    import random

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=HF_DATASET_NAME,
        repo_type="dataset",
        revision="refs/convert/parquet",
        filename=_PARQUET_SHARD[hf_split],
        token=HF_TOKEN,
    )

    parquet_file = pq.ParquetFile(local_path)
    batch = next(parquet_file.iter_batches(batch_size=limit))
    rows = batch.to_pylist()

    for row in rows:
        row["image"] = Image.open(io.BytesIO(row["image"]["bytes"]))

    random.Random(seed).shuffle(rows)
    return rows


def _save_image(pil_image, path: Path) -> None:
    if not path.exists():
        pil_image.convert("RGB").save(path, format="JPEG", quality=90)


def _voc_bbox_to_xywh(bbox: list[float]) -> tuple[float, float, float, float]:
    """Converte bbox de Pascal VOC [x_min,y_min,x_max,y_max] (formato original
    do Fashionpedia) para [x_min,y_min,width,height]. Usado tanto pelo export
    COCO (pixels absolutos) quanto pelo YOLO (que ainda normaliza por cima)."""
    x_min, y_min, x_max, y_max = bbox
    return x_min, y_min, x_max - x_min, y_max - y_min


def convert_split_to_coco(examples: Iterable[dict], category_names: list[str],
                           images_dir: Path, ann_path: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 0

    for example in examples:
        image_id = example["image_id"]
        width, height = example["width"], example["height"]
        file_name = f"{image_id}.jpg"
        _save_image(example["image"], images_dir / file_name)

        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

        objects = example["objects"]
        for bbox, category, area in zip(objects["bbox"], objects["category"], objects["area"]):
            x_min, y_min, box_w, box_h = _voc_bbox_to_xywh(bbox)
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category,
                "bbox": [x_min, y_min, box_w, box_h],
                "area": float(area),
                "iscrowd": 0,
            })
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name, "supercategory": "fashion"} for i, name in enumerate(category_names)],
    }
    with open(ann_path, "w") as f:
        json.dump(coco_dict, f)


def convert_split_to_yolo(examples: Iterable[dict], images_dir: Path, labels_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for example in examples:
        image_id = example["image_id"]
        width, height = example["width"], example["height"]
        _save_image(example["image"], images_dir / f"{image_id}.jpg")

        lines = []
        objects = example["objects"]
        for bbox, category in zip(objects["bbox"], objects["category"]):
            x_min, y_min, box_w, box_h = _voc_bbox_to_xywh(bbox)
            x_center = (x_min + box_w / 2) / width
            y_center = (y_min + box_h / 2) / height
            lines.append(f"{category} {x_center:.6f} {y_center:.6f} {box_w / width:.6f} {box_h / height:.6f}")

        (labels_dir / f"{image_id}.txt").write_text("\n".join(lines))


def write_yolo_yaml(yolo_dir: Path, category_names: list[str]) -> Path:
    yaml_path = yolo_dir / "data.yaml"
    lines = [
        f"path: {yolo_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ] + [f"  {i}: {name}" for i, name in enumerate(category_names)]
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


class CocoDetectionDataset(TorchDataset):
    """Lê um `instances_*.json` (formato COCO, gerado por `convert_fashionpedia`)
    e devolve (imagem, target) no formato esperado pelo torchvision:
    boxes em [x1, y1, x2, y2] absolutos e labels deslocados +1 (0 é a classe
    de background reservada pelo Faster R-CNN / SSD do torchvision).
    Usada por `src/train.py` e `src/eval.py`.
    """

    def __init__(self, images_dir: str | Path, ann_path: str | Path):
        self.images_dir = Path(images_dir)

        with open(ann_path) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        self.annotations_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image = Image.open(self.images_dir / image_info["file_name"]).convert("RGB")
        image = TF.to_tensor(image)

        anns = self.annotations_by_image.get(image_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"] + 1)

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([image_id])}
        return image, target


def detection_collate_fn(batch):
    """collate_fn para DataLoader com CocoDetectionDataset: imagens têm
    tamanhos diferentes, então não dá pra empilhar em um único tensor."""
    return tuple(zip(*batch))


def convert_fashionpedia(output_dir: str | Path = "data", val_ratio: float = 0.1,
                          limit: int | None = None, seed: int = 42,
                          formats: Iterable[str] = ("coco", "yolo")) -> dict[str, int]:
    """Baixa o Fashionpedia e converte para os formatos pedidos em `formats`.

    Se `limit` for informado, baixa só 1 shard parquet pequeno e usa uma
    amostra de `limit` exemplos dele (usado pelo smoke test, para não
    precisar baixar o dataset inteiro). Retorna a quantidade de imagens
    geradas em cada split.
    """
    output_dir = Path(output_dir)
    category_names = get_category_names()

    if limit is None:
        split = _load_split_full("train").train_test_split(test_size=val_ratio, seed=seed)
        train_examples, val_examples = split["train"], split["test"]
        test_examples = _load_split_full("val")
    else:
        # min(..., limit - 1) garante pelo menos 1 exemplo de treino quando
        # `limit` é pequeno (ex: limit=1 sem essa trava deixaria o split de
        # treino vazio, já que val_count poderia ser igual a limit).
        val_count = max(1, min(int(limit * val_ratio), limit - 1)) if limit > 1 else 0
        examples = _load_split_limited("train", limit, seed)
        train_examples, val_examples = examples[val_count:], examples[:val_count]
        test_examples = _load_split_limited("val", max(1, limit // 5), seed)

    splits = {"train": train_examples, "val": val_examples, "test": test_examples}

    if "coco" in formats:
        for name, examples in splits.items():
            convert_split_to_coco(
                examples, category_names,
                images_dir=output_dir / "coco" / "images" / name,
                ann_path=output_dir / "coco" / "annotations" / f"instances_{name}.json",
            )

    if "yolo" in formats:
        for name, examples in splits.items():
            convert_split_to_yolo(
                examples,
                images_dir=output_dir / "yolo" / "images" / name,
                labels_dir=output_dir / "yolo" / "labels" / name,
            )
        write_yolo_yaml(output_dir / "yolo", category_names)

    return {name: len(examples) for name, examples in splits.items()}


def run_eda(output_dir: str | Path = "data", coco_ann_path: str | Path | None = None,
            split: str = "train", save_plot: bool = True, plots_dir: str | Path = "logs") -> dict:
    """Mostra a distribuição de classes e o tamanho médio das bounding boxes de um
    split já convertido para COCO (rode `convert_fashionpedia` antes).
    """
    output_dir = Path(output_dir)
    ann_path = Path(coco_ann_path) if coco_ann_path else output_dir / "coco" / "annotations" / f"instances_{split}.json"

    with open(ann_path) as f:
        coco = json.load(f)

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    class_counts = Counter(ann["category_id"] for ann in coco["annotations"])
    widths = defaultdict(list)
    heights = defaultdict(list)
    for ann in coco["annotations"]:
        _, _, w, h = ann["bbox"]
        widths[ann["category_id"]].append(w)
        heights[ann["category_id"]].append(h)

    stats = {
        id_to_name[cat_id]: {
            "count": count,
            "avg_bbox_width": sum(widths[cat_id]) / len(widths[cat_id]),
            "avg_bbox_height": sum(heights[cat_id]) / len(heights[cat_id]),
        }
        for cat_id, count in class_counts.items()
    }

    print(f"EDA — split '{split}' ({len(coco['images'])} imagens, {len(coco['annotations'])} bboxes)")
    print(f"{'classe':<25}{'qtd':>8}{'largura_media':>16}{'altura_media':>16}")
    for name, s in sorted(stats.items(), key=lambda kv: -kv[1]["count"]):
        print(f"{name:<25}{s['count']:>8}{s['avg_bbox_width']:>16.1f}{s['avg_bbox_height']:>16.1f}")

    if save_plot:
        import matplotlib.pyplot as plt

        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        names_sorted = sorted(stats, key=lambda n: -stats[n]["count"])
        counts_sorted = [stats[n]["count"] for n in names_sorted]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(names_sorted, counts_sorted)
        ax.set_ylabel("Número de bounding boxes")
        ax.set_title(f"Distribuição de classes — split '{split}'")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig_path = plots_dir / f"eda_class_distribution_{split}.png"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Gráfico salvo em {fig_path}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte o Fashionpedia para COCO/YOLO e roda EDA.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None, help="Usa só N exemplos (streaming), para testes rápidos")
    parser.add_argument("--formats", nargs="+", default=["coco", "yolo"])
    parser.add_argument("--eda", action="store_true")
    args = parser.parse_args()

    counts = convert_fashionpedia(
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        limit=args.limit,
        formats=args.formats,
    )
    print("Exemplos convertidos por split:", counts)

    if args.eda:
        run_eda(output_dir=args.output_dir, split="train")
