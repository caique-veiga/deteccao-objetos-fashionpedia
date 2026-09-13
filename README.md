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

## Como reproduzir

1. Criar e ativar o ambiente virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Criar um `.env` na raiz do projeto com:
   ```
   HF_TOKEN=seu_token_do_huggingface_aqui
   ```

3. Rodar o smoke test local (CPU, poucos dados) antes de qualquer treino real:
   ```bash
   python tests/smoke_test.py
   ```

4. Treino completo (recomendado no Google Colab com GPU): abrir
   `notebooks/colab_orchestrator.ipynb`, que clona este repositório, instala as
   dependências e chama as funções de `src/train.py` e `src/eval.py`.

## Metodologia

- **Dataset:** Fashionpedia (Hugging Face `datasets`), anotações convertidas para o
  formato COCO (train/val/test) e para o formato YOLO (`.txt`).
- **Treino:** cada modelo é treinado separadamente, com checkpoint salvo a cada época
  e possibilidade de retomar treino interrompido.
- **Avaliação:** predições exportadas no formato COCO results JSON e avaliadas com
  `pycocotools.cocoeval.COCOeval` (mAP@0.5 e mAP@0.5:0.95).
- **Comparação:** gráficos de loss (train vs val) por modelo e gráfico de barras
  comparando mAP dos três modelos lado a lado.

## Resultados

_(preencher após os experimentos: métricas de mAP, curvas de loss, exemplos visuais de
acertos/erros de cada modelo, conclusão comparativa)_
