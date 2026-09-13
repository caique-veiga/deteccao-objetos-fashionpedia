# Detecção de Objetos em Moda — YOLOv8 vs Faster R-CNN vs SSD

Projeto de pós-graduação (PUC-RJ, disciplina de Visão Computacional) que compara três
arquiteturas de detecção de objetos no dataset **Fashionpedia** (Hugging Face):

- **YOLOv8** (ultralytics)
- **Faster R-CNN** (torchvision, ResNet-50 FPN)
- **SSD300** (torchvision, VGG16 backbone)

A avaliação é feita com `pycocotools`, reportando **mAP@0.5** e **mAP@0.5:0.95** para
cada modelo, além de curvas de train/val loss por época.

## Estrutura do projeto

```
├── data/                  # dados baixados (não versionado)
├── notebooks/
│   └── colab_orchestrator.ipynb   # orquestra o treino/avaliação no Colab (GPU)
├── src/
│   ├── data.py            # download Fashionpedia + conversão para COCO/YOLO + EDA
│   ├── models.py          # construção dos 3 modelos (YOLO, Faster R-CNN, SSD)
│   ├── train.py           # loops de treino com logging de loss por época
│   ├── eval.py            # inferência + cálculo de mAP via pycocotools
│   └── plotting.py        # gráficos de loss e comparação de mAP
├── tests/
│   └── smoke_test.py      # teste rápido em CPU com poucos dados, antes do Colab
├── checkpoints/           # pesos salvos (não versionado)
├── logs/                  # CSVs de loss por época (versionado, são leves)
├── .env                   # tokens/chaves (não versionado)
├── requirements.txt
└── README.md
```

## Status atual

Pipeline completo implementado e validado localmente (CPU, poucas imagens, via
`tests/smoke_test.py`). O treino completo (dataset inteiro, GPU) ainda não foi
rodado — é o próximo passo, no Google Colab. A seção [Resultados](#resultados)
será preenchida depois disso.

## Como reproduzir

1. Criar e ativar o ambiente virtual, e instalar as dependências:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Criar um `.env` na raiz do projeto com:
   ```
   HF_TOKEN=seu_token_do_huggingface_aqui
   ```

3. Rodar o smoke test local (CPU, ~20 imagens, 1 época) antes de qualquer treino
   real — garante que `data.py`, `models.py`, `train.py` e `eval.py` funcionam
   sem erro antes de gastar tempo de GPU:
   ```bash
   python tests/smoke_test.py
   ```

4. Converter o dataset completo (baixa o Fashionpedia inteiro e gera os
   formatos COCO e YOLO em `data/`):
   ```bash
   python src/data.py --output-dir data --eda
   ```

5. Treinar cada modelo (mesmo script para os 3, batch size/épocas/patience
   compartilhados para manter a comparação justa — ver nota em `src/train.py`):
   ```bash
   python src/train.py fasterrcnn --epochs 30 --batch-size 8 --patience 5
   python src/train.py ssd        --epochs 30 --batch-size 8 --patience 5
   python src/train.py yolo       --epochs 30 --batch-size 8 --patience 5
   ```
   Para retomar um treino interrompido: `--resume-from checkpoints/fasterrcnn_epoch12.pt`
   (fasterrcnn/ssd) ou `--resume` (yolo).

6. Avaliar cada modelo no conjunto de teste (mAP via `pycocotools`):
   ```bash
   python src/eval.py fasterrcnn --checkpoint checkpoints/fasterrcnn_epoch29.pt
   python src/eval.py ssd        --checkpoint checkpoints/ssd_epoch29.pt
   python src/eval.py yolo       --checkpoint checkpoints/yolo/weights/best.pt
   ```

7. Gerar os gráficos finais:
   ```bash
   python src/plotting.py loss fasterrcnn
   python src/plotting.py loss ssd
   python src/plotting.py loss yolo
   python src/plotting.py map --metrics-path logs/metrics.json
   ```

Os passos 4–7 rodam melhor no **Google Colab** (GPU), clonando este repositório
e instalando `requirements.txt` — ver `notebooks/colab_orchestrator.ipynb`
(a ser criado diretamente no Colab, com células markdown documentando cada
etapa do experimento, conforme exigido no enunciado do trabalho).

## Metodologia

- **Dataset:** Fashionpedia (Hugging Face `datasets`, 46 categorias). O HF só
  disponibiliza os splits `train`/`val`; dividimos `train` em train/val para uso
  durante o treino, e usamos o `val` original como nosso conjunto de teste
  (nunca visto durante o treino/tuning).
- **Treino:** cada modelo é treinado separadamente, com checkpoint salvo a cada
  época e possibilidade de retomar treino interrompido. Batch size, número de
  épocas e critério de early stopping (`--patience`) são os mesmos nos três
  modelos; Faster R-CNN/SSD usam SGD com decaimento cosseno de LR (o YOLO já
  decai o LR automaticamente via ultralytics). Data augmentation **não** foi
  equalizada: o YOLO usa as augmentations padrão do ultralytics (mosaic, flip,
  HSV jitter), Faster R-CNN/SSD não aplicam nenhuma — limitação conhecida,
  discutida na conclusão.
- **Avaliação:** predições exportadas no formato COCO results JSON e avaliadas com
  `pycocotools.cocoeval.COCOeval` (mAP@0.5 e mAP@0.5:0.95).
- **Comparação:** gráficos de loss (train vs val) por modelo e gráfico de barras
  comparando mAP dos três modelos lado a lado.

## Resultados

_(preencher após os experimentos no Colab: métricas de mAP, curvas de loss,
exemplos visuais de acertos/erros de cada modelo, conclusão comparativa)_
