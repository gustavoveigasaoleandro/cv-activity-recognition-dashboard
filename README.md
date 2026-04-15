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
