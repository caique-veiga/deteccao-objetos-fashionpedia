"""Loops de treino para os 3 modelos comparados neste projeto.

Faster R-CNN e SSD (torchvision) compartilham a mesma API de forward
(`model(images, targets)` retorna um dict de losses em modo `.train()`),
então usamos um único loop de treino genérico, `train_torchvision_model`,
para os dois. Para o YOLO, delegamos ao treino nativo do ultralytics
(`train_yolo_model`), que já gera seus próprios logs de loss por época e
salva checkpoints — não faz sentido reimplementar isso.

Nota sobre "val loss" nos modelos torchvision: esses modelos só retornam o
dict de losses em modo `.train()` (em `.eval()` retornam predições, não
losses). Por isso calculamos a "val loss" também com o modelo em modo
`.train()`, mas dentro de `torch.no_grad()` e sem chamar `optimizer.step()`
— é uma limitação conhecida da API de detecção do torchvision, não um erro.

Nota sobre comparação justa entre os modelos: o ultralytics já treina o
YOLO com otimizador/LR escolhidos automaticamente (`optimizer=auto`, hoje
resolve pra AdamW) e com decaimento de LR embutido. Faster R-CNN e SSD, por
não terem esse recurso pronto no torchvision, usam SGD com um cosine decay
adicionado manualmente aqui — critério aproximado, não idêntico ao do YOLO,
mas evita deixá-los treinando com LR constante o tempo todo enquanto o YOLO
já decai o dele. Batch size, número de épocas e "patience" (early stopping)
são equivalentes nos três: o loop torchvision não tem early stopping, então
`train_yolo_model` desabilita o do YOLO (`patience=epochs`) pra manter a
mesma condição. Data augmentation NÃO foi equalizada: o YOLO usa as
augmentations padrão do ultralytics (mosaic, flip, HSV jitter), enquanto o
`CocoDetectionDataset` usado por Faster R-CNN/SSD não aplica nenhuma — vale
mencionar isso como limitação no relatório final.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import CocoDetectionDataset, detection_collate_fn


def _cosine_lr_lambda(epoch: int, total_epochs: int) -> float:
    """Fator multiplicativo do LR base, decaindo em cosseno até ~0 na
    última época de `total_epochs`. Forma fechada (só depende de `epoch` e
    `total_epochs`, sem estado recursivo) para dar continuidade correta ao
    retomar treino de um checkpoint — ver `train_torchvision_model`.
    """
    return 0.5 * (1 + math.cos(math.pi * epoch / max(total_epochs - 1, 1)))


def train_torchvision_model(
    model,
    train_dataset: CocoDetectionDataset,
    val_dataset: CocoDetectionDataset,
    model_name: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.005,
    device: str | None = None,
    checkpoints_dir: str | Path = "checkpoints",
    logs_dir: str | Path = "logs",
    resume_from: str | Path | None = None,
    patience: int | None = None,
) -> Path:
    """Treina um modelo torchvision (Faster R-CNN ou SSD).

    Registra train/val loss por época em `logs/{model_name}_loss.csv` e
    salva um checkpoint por época em `checkpoints/{model_name}_epoch{N}.pt`.
    Se `resume_from` apontar para um checkpoint existente, carrega pesos e
    otimizador dele e continua a contagem de épocas a partir daí.

    Se `patience` for informado, o treino para mais cedo caso a val_loss não
    melhore por `patience` épocas seguidas (mesmo critério de early stopping
    do `train_yolo_model`, pra manter os 3 modelos na mesma condição).

    Retorna o caminho do último checkpoint salvo.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoints_dir = Path(checkpoints_dir)
    logs_dir = Path(logs_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    start_epoch = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # restaura o lr salvo, ignora `lr` abaixo
        start_epoch = checkpoint["epoch"] + 1
        print(f"[{model_name}] retomando treino de '{resume_from}' a partir da época {start_epoch} "
              f"(parâmetro lr={lr} ignorado: o otimizador retomado já traz o lr salvo no checkpoint)")

    # Cosine annealing: o ultralytics já decai o LR automaticamente no YOLO,
    # então adicionamos um schedule aqui pra não deixar Faster R-CNN/SSD em
    # desvantagem por rodar com LR constante o treino inteiro.
    #
    # Usamos LambdaLR com uma fórmula fechada (só depende de `epoch` e do
    # total de épocas) em vez de CosineAnnealingLR: o CosineAnnealingLR do
    # PyTorch é recursivo internamente (o LR de uma época depende do LR da
    # anterior), então reconstruí-lo do zero ao retomar treino (com
    # `last_epoch` diferente de -1) gera um LR incorreto logo de cara —
    # confirmado testando isoladamente antes de usar aqui. Com forma
    # fechada, retomar com `last_epoch=start_epoch-1` estende a curva de
    # forma consistente com o que teria acontecido numa execução contínua.
    total_epochs = start_epoch + epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _cosine_lr_lambda(epoch, total_epochs),
        last_epoch=start_epoch - 1,
    )

    log_path = logs_dir / f"{model_name}_loss.csv"
    if resume_from is None or not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "duration_sec"])

    last_checkpoint_path = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(start_epoch, start_epoch + epochs):
        start_time = time.time()

        model.train()
        train_loss_sum = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
        train_loss = train_loss_sum / max(len(train_loader), 1)

        model.train()  # losses só existem em modo train(); ver nota no topo do arquivo
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss_sum += sum(loss_dict.values()).item()
        val_loss = val_loss_sum / max(len(val_loader), 1)

        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - start_time
        print(f"[{model_name}] época {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={current_lr:.6f} ({duration:.1f}s)")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, current_lr, round(duration, 1)])

        scheduler.step()

        last_checkpoint_path = checkpoints_dir / f"{model_name}_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, last_checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Nota: `best_val_loss`/`epochs_without_improvement` não persistem
        # entre chamadas (não são salvos no checkpoint) — um resume reinicia
        # a contagem de paciência do zero, assim como o schedule de LR
        # (ver comentário acima sobre `total_epochs`).
        if patience is not None and epochs_without_improvement >= patience:
            print(f"[{model_name}] early stopping: sem melhora na val_loss por {patience} épocas seguidas")
            break

    return last_checkpoint_path


def train_yolo_model(
    yolo_model,
    data_yaml: str | Path,
    epochs: int = 10,
    batch_size: int = 4,
    imgsz: int = 640,
    project: str | Path = "checkpoints",
    name: str = "yolo",
    resume: bool = False,
    patience: int | None = None,
):
    """Treina um modelo YOLO via ultralytics.

    O ultralytics já salva checkpoints (`best.pt` / `last.pt`) em
    `{project}/{name}/weights/` e gera um `results.csv` com o loss por
    época automaticamente — não precisamos reimplementar esse logging.

    Para retomar um treino interrompido, passe `resume=True`: o ultralytics
    usa automaticamente o `last.pt` salvo em `{project}/{name}/weights/`.

    Usamos caminho absoluto em `project` porque o ultralytics resolve
    caminhos relativos contra o `runs_dir` das configurações globais dele
    (`~/.config/Ultralytics/settings.json`), não contra o diretório atual —
    isso pode jogar a saída pra fora da pasta do projeto dependendo do que
    estiver configurado na máquina.

    `patience`: mesmo critério de early stopping usado em
    `train_torchvision_model` (para depois de `patience` épocas sem melhora
    na val_loss), pra manter os 3 modelos na mesma condição. Se `None`
    (padrão), usamos `patience=epochs`, o que garante que o early stopping
    do ultralytics nunca dispare dentro da execução — equivalente a "sem
    early stopping", igual ao comportamento padrão do loop torchvision.
    """
    return yolo_model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience if patience is not None else epochs,
        project=str(Path(project).resolve()),
        name=name,
        resume=resume,
    )


if __name__ == "__main__":
    import argparse

    import models

    parser = argparse.ArgumentParser(description="Treina um dos 3 modelos de detecção.")
    parser.add_argument("model", choices=["fasterrcnn", "ssd", "yolo"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005, help="Só usado por fasterrcnn/ssd")
    parser.add_argument("--resume-from", default=None, help="Checkpoint (.pt) pra retomar (fasterrcnn/ssd)")
    parser.add_argument("--resume", action="store_true", help="Retoma treino do last.pt (yolo)")
    parser.add_argument("--patience", type=int, default=None,
                         help="Early stopping: para após N épocas sem melhora na val_loss. "
                              "Sem essa flag, nenhum dos 3 modelos usa early stopping.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.model in ("fasterrcnn", "ssd"):
        import data as data_module

        num_classes = len(data_module.get_category_names())
        model = models.get_fasterrcnn_model(num_classes) if args.model == "fasterrcnn" else models.get_ssd_model(num_classes)

        train_ds = CocoDetectionDataset(
            data_dir / "coco" / "images" / "train", data_dir / "coco" / "annotations" / "instances_train.json"
        )
        val_ds = CocoDetectionDataset(
            data_dir / "coco" / "images" / "val", data_dir / "coco" / "annotations" / "instances_val.json"
        )

        train_torchvision_model(
            model, train_ds, val_ds, model_name=args.model,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            resume_from=args.resume_from, patience=args.patience,
        )
    else:
        yolo = models.get_yolo_model()
        train_yolo_model(
            yolo, data_dir / "yolo" / "data.yaml",
            epochs=args.epochs, batch_size=args.batch_size, resume=args.resume,
            patience=args.patience,
        )
