"""Construção dos três modelos de detecção comparados neste projeto:
YOLOv8 (ultralytics), Faster R-CNN e SSD300 (torchvision).

Convenção de `num_classes`: em `get_fasterrcnn_model` e `get_ssd_model`,
`num_classes` é o número de classes de objeto reais (46 no Fashionpedia).
Internamente somamos 1 para a classe de fundo ("background"), que o
torchvision exige para esses dois modelos. O YOLO não usa essa convenção
(ultralytics não tem classe de fundo explícita) — o número de classes é
lido automaticamente do `data.yaml` na hora do treino.
"""

from __future__ import annotations

import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    ssd300_vgg16,
    SSD300_VGG16_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead


def _retrieve_out_channels(backbone, image_size: tuple[int, int]) -> list[int]:
    """Descobre o número de canais de saída de cada feature map do backbone,
    rodando um forward de teste com um tensor aleatório.

    Implementado localmente (em vez de importar
    `torchvision.models.detection._utils.retrieve_out_channels`, que faz
    exatamente isso) porque é uma API privada — o nome/local pode mudar sem
    aviso entre versões do torchvision, quebrando `get_ssd_model` sem
    alternativa pronta.
    """
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        device = next(backbone.parameters()).device
        dummy_image = torch.zeros((1, 3, *image_size), device=device)
        features = backbone(dummy_image)
    backbone.train(was_training)

    features = [features] if isinstance(features, torch.Tensor) else list(features.values())
    return [f.shape[1] for f in features]


def get_yolo_model(model_name: str = "yolov8n.pt"):
    """Carrega um modelo YOLOv8 (pesos pré-treinados na COCO por padrão).

    `model_name` pode ser qualquer checkpoint do ultralytics (ex: yolov8n.pt,
    yolov8s.pt, yolov8m.pt) ou um caminho para um checkpoint próprio (para
    retomar treino). O número de classes é definido depois, no treino, via
    o `data.yaml` gerado por `src/data.py`.
    """
    from ultralytics import YOLO

    return YOLO(model_name)


def get_fasterrcnn_model(num_classes: int, pretrained: bool = True):
    """Cria um Faster R-CNN (ResNet-50 FPN) com a head de classificação
    substituída para `num_classes` classes de objeto (+ 1 background).
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model


def get_ssd_model(num_classes: int, pretrained: bool = True):
    """Cria um SSD300 (backbone VGG16) com a head de classificação
    substituída para `num_classes` classes de objeto (+ 1 background).
    """
    weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
    model = ssd300_vgg16(weights=weights)

    in_channels = _retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1,
    )

    return model


if __name__ == "__main__":
    NUM_CLASSES = 46  # categorias do Fashionpedia

    print("Carregando YOLOv8...")
    yolo = get_yolo_model()
    print("  OK:", type(yolo).__name__)

    print("Carregando Faster R-CNN...")
    fasterrcnn = get_fasterrcnn_model(NUM_CLASSES)
    out_features = fasterrcnn.roi_heads.box_predictor.cls_score.out_features
    print(f"  OK: head com {out_features} saídas ({NUM_CLASSES} classes + background)")

    print("Carregando SSD300...")
    ssd = get_ssd_model(NUM_CLASSES)
    print("  OK: modelo construído,", type(ssd.head.classification_head).__name__)
