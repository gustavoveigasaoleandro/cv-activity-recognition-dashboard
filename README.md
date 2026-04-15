# cv-activity-recognition-dashboard

Projeto de visao computacional com foco em classificacao de atividades humanas e deteccao de queda usando TensorFlow e um dashboard em Dash para inferencia por imagem.

## O que o repositorio contem

- `preparing_ds.py`: pipeline de treinamento para classificacao multiclasse de acoes humanas com EfficientNetV2B0.
- `preparing_ds2.py`: experimento separado para classificacao de quedas a partir de um dataset estruturado em imagens e labels.
- `dash/`: interface web simples para carregar uma imagem e visualizar a classe mais provavel e o top 3 de previsoes.
- `best_by_f1.keras`: checkpoint final utilizado pelo dashboard.
- `Checkpoint/F1checkpoint.py`: callback customizado de treinamento.

## O que foi removido

Para manter o repositorio leve e seguro, esta versao nao inclui:

- datasets brutos (`Human_Action_Recognition/` e `fall_dataset/`);
- ambiente virtual local;
- arquivos de submissao e predicoes geradas;
- artefatos auxiliares que nao sao necessarios para leitura do projeto.

Nao foram encontrados segredos, tokens ou chaves de API no material publicado.

## Bases de dados utilizadas

O projeto foi estruturado para trabalhar com duas bases distintas, ambas removidas desta versao publicada.

### 1. Base de classificacao de acoes humanas

Consumida por `preparing_ds.py`, esta base e esperada no diretorio `Human_Action_Recognition/` com a seguinte estrutura:

- `Training_set.csv`: arquivo de treino com colunas `filename` e `label`
- `Testing_set.csv`: arquivo de teste com coluna `filename`
- `train/`: imagens usadas no treinamento
- `test/`: imagens usadas para inferencia/submissao

No pipeline, os nomes das classes sao obtidos a partir das labels presentes em `Training_set.csv`, e o modelo treina uma classificacao multiclasse de acoes humanas com `EfficientNetV2B0`.

### 2. Base de deteccao/classificacao de queda

Consumida por `preparing_ds2.py`, esta base e esperada no diretorio `fall_dataset/` com a seguinte estrutura:

- `images/train`
- `images/val`
- `labels/train`
- `labels/val`

As anotacoes sao lidas a partir dos arquivos de label e convertidas para tres classes fixas no codigo:

- `falling_person`
- `lying_person`
- `standing_person`

Esse segundo pipeline trata o problema como classificacao de estado/atividade relacionada a queda, reutilizando `EfficientNetV2B0` como backbone.

### Origem e licenca

O material original deste projeto nao registra com precisao o nome oficial, o link de origem ou a licenca dessas duas bases. Por isso, esta documentacao descreve apenas a estrutura e o uso observados no codigo, sem atribuir uma fonte externa que nao esteja comprovada no repositorio.

## Como executar o dashboard

1. Instale as dependencias:

```bash
pip install -r requirements.txt
```

2. Inicie a interface:

```bash
cd dash
python app.py
```

O dashboard usa `best_by_f1.keras`. Se `Human_Action_Recognition/Training_set.csv` nao estiver presente, a interface ainda funciona, mas exibira nomes genericos de classe (`class_0`, `class_1`, etc.).

## Como reutilizar os scripts de treino

Os scripts assumem que os datasets existam localmente nestes caminhos:

- `Human_Action_Recognition/`
- `fall_dataset/`

Se voce quiser reproduzir o treinamento, recoloque os datasets nesses diretorios antes de executar os scripts.
