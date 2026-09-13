# Guia Passo a Passo — Projeto de Detecção de Objetos (YOLO vs Faster R-CNN vs SSD)

Workflow: **VSCode + Claude Code (estrutura o código) → GitHub (versiona) → Google Colab (treina com GPU)**

---

## 1. Configurar o VSCode

Extensões para instalar (Ctrl+Shift+X):
- **Python** (Microsoft)
- **Jupyter** (Microsoft) — permite rodar/testar células localmente se quiser
- **Claude Code** (extensão oficial, ou use via terminal integrado com `claude`)
- **GitLens** (opcional, ajuda a visualizar histórico de commits)

Criar e ativar um ambiente virtual (terminal integrado do VSCode, `Ctrl+\``):

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

Selecionar o interpretador: `Ctrl+Shift+P` → "Python: Select Interpreter" → escolher o `venv` criado.

---

## 2. Estrutura de pastas do projeto

Peça isso pro Claude Code criar (prompt na seção 4), mas o resultado esperado é:

```
projeto-deteccao/
├── data/                  # dados baixados (NÃO vai pro git)
├── notebooks/
│   └── colab_orchestrator.ipynb
├── src/
│   ├── data.py            # conversão HuggingFace -> formato COCO
│   ├── models.py          # wrappers YOLO, Faster R-CNN, SSD
│   ├── train.py           # loop de treino + logging train/val loss
│   ├── eval.py            # avaliação com pycocotools (mAP)
│   └── plotting.py        # gráficos de loss e comparação de mAP
├── checkpoints/           # pesos salvos (NÃO vai pro git)
├── logs/                  # CSVs de loss por época (pode versionar, são leves)
├── .env                   # tokens/chaves (NÃO vai pro git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 3. Inicializar o Git corretamente

No terminal do VSCode, dentro da pasta do projeto:

```bash
git init
git branch -M main
```

### `.gitignore` — crie este arquivo com este conteúdo:

```gitignore
# Ambiente virtual
venv/
env/
__pycache__/
*.pyc

# Variáveis de ambiente e segredos
.env
*.key

# Dados (geralmente grandes demais pro git, e nem deveriam ir)
data/
*.zip
*.tar.gz

# Modelos e checkpoints (grandes, não pertencem ao git)
checkpoints/
*.pt
*.pth
*.onnx
runs/
wandb/

# Jupyter
.ipynb_checkpoints/

# Outputs de treino grandes
outputs/
predictions.json

# Sistema
.DS_Store
Thumbs.db

# VSCode
.vscode/
```

### `.env` — crie este arquivo (exemplo, ajuste conforme suas necessidades):

```bash
HF_TOKEN=seu_token_do_huggingface_aqui
WANDB_API_KEY=sua_chave_wandb_se_for_usar
```

> **Importante:** o `.env` só existe na sua máquina/Colab, nunca no GitHub. Para usar essas variáveis no código, instale `python-dotenv`:
> ```bash
> pip install python-dotenv
> ```
> E no início dos seus scripts:
> ```python
> from dotenv import load_dotenv
> import os
> load_dotenv()
> token = os.getenv("HF_TOKEN")
> ```

### Verificação antes do primeiro commit (passo que muita gente pula e se arrepende):

```bash
git status
```
Confira que `data/`, `checkpoints/`, `.env` e `venv/` **não aparecem** como arquivos a serem commitados. Se aparecerem, o `.gitignore` não está funcionando corretamente — revise antes de continuar.

---

## 4. Prompts para usar com o Claude Code, na ordem

Rode `claude` no terminal do VSCode dentro da pasta do projeto, ou use a extensão. Sugestão de prompts, um de cada vez, revisando o resultado antes de passar pro próximo:

**Prompt 1 — Estrutura inicial:**
> "Crie a estrutura de pastas do projeto conforme: data/, notebooks/, src/, checkpoints/, logs/. Crie também um requirements.txt com torch, torchvision, ultralytics, pycocotools, matplotlib, python-dotenv, datasets (huggingface) e um README.md inicial descrevendo o projeto: comparação de YOLOv8, Faster R-CNN e SSD para detecção de objetos no dataset Fashionpedia, calculando mAP com pycocotools."

**Prompt 2 — Conversão de dados:**
> "Em src/data.py, escreva uma função que carrega o dataset Fashionpedia do HuggingFace usando a biblioteca datasets, e converte as anotações (bounding boxes) para o formato COCO json (train/val/test), e também para o formato YOLO txt. Inclua uma função de EDA que mostra distribuição de classes e tamanho médio dos bounding boxes."

**Prompt 3 — Modelos:**
> "Em src/models.py, crie três funções: get_yolo_model() usando ultralytics YOLOv8, get_fasterrcnn_model(num_classes) usando torchvision.models.detection.fasterrcnn_resnet50_fpn com a head substituída para o número de classes do dataset, e get_ssd_model(num_classes) usando torchvision.models.detection.ssd300_vgg16, também com a head ajustada."

**Prompt 4 — Treino com logging:**
> "Em src/train.py, escreva loops de treino separados para Faster R-CNN e SSD (torchvision) que registram train loss e val loss por época em um CSV dentro de logs/. Para o YOLO, escreva a chamada de treino via ultralytics que já gera esse log automaticamente. Adicione salvamento de checkpoint a cada época em checkpoints/, e um parâmetro para retomar treino a partir de um checkpoint existente."

**Prompt 5 — Avaliação com mAP:**
> "Em src/eval.py, escreva uma função que roda inferência de qualquer um dos 3 modelos no conjunto de teste, exporta as predições no formato COCO results json, e usa pycocotools (COCO, COCOeval) para calcular mAP@0.5 e mAP@0.5:0.95. A função deve retornar um dicionário com essas métricas para cada modelo, para eu poder comparar depois."

**Prompt 6 — Gráficos:**
> "Em src/plotting.py, escreva uma função plot_loss_curves(model_name) que lê o CSV de logs/ e plota train loss vs val loss por época, salvando como PNG. Escreva também plot_map_comparison(results_dict) que gera um gráfico de barras comparando mAP@0.5 e mAP@0.5:0.95 dos três modelos lado a lado."

**Prompt 7 — Teste local antes de gastar GPU:**
> "Crie um script tests/smoke_test.py que roda 1 época com um subconjunto pequeno de dados (ex: 20 imagens) na CPU, para os três modelos, só para garantir que data.py, models.py, train.py e eval.py funcionam sem erro antes de eu rodar no Colab de verdade."

Rode `python tests/smoke_test.py` localmente. Só depois de passar sem erros, siga para o Colab.

---

## 5. Subir pro GitHub

```bash
git add .
git status                    # confira de novo o que vai subir
git commit -m "Setup inicial: estrutura, data pipeline, modelos, treino e avaliação"
```

Crie um repositório vazio no GitHub (github.com → New repository, **não** inicialize com README/gitignore, já temos os nossos), depois:

```bash
git remote add origin https://github.com/SEU_USUARIO/SEU_REPO.git
git push -u origin main
```

> Se o repo for público, faça uma última checagem manual olhando o repositório no navegador depois do push, confirmando que não subiu nada sensível.

---

## 6. Rodar no Colab

No Colab, célula 1:
```python
!git clone https://github.com/SEU_USUARIO/SEU_REPO.git
%cd SEU_REPO
!pip install -r requirements.txt
```

Célula 2 — recriar o `.env` manualmente no Colab (ele não vem do GitHub, de propósito):
```python
%%writefile .env
HF_TOKEN=seu_token_aqui
```

Célula 3 — montar o Drive para salvar checkpoints/logs persistentes:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Depois disso, as células seguintes chamam as funções de `src/train.py` e `src/eval.py`, com células de markdown intercaladas explicando cada etapa (exigido no enunciado do trabalho).

---

## 7. Checklist final antes de considerar pronto

- [ ] `.env` e `data/` nunca aparecem no `git status`
- [ ] `requirements.txt` está atualizado com tudo que foi usado
- [ ] Os 3 modelos têm curva de train/val loss plotada
- [ ] mAP@0.5 e mAP@0.5:0.95 calculados para os 3 modelos com pycocotools
- [ ] Gráfico comparativo final dos 3 modelos
- [ ] Exemplos visuais de acertos/erros de cada modelo
- [ ] README.md explica como reproduzir o experimento
- [ ] Notebook final tem células markdown para: problema, base de dados, metodologia, experimentos, resultados e conclusão
