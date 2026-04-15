# cv-activity-recognition-dashboard

Projeto de visao computacional para classificacao de atividades humanas e estudo de deteccao de quedas, com treinamento em TensorFlow e dashboard Dash para inferencia por imagem.

## Conteudo

- `preparing_ds.py`: pipeline de treinamento para classificacao multiclasse de acoes humanas com EfficientNetV2B0.
- `preparing_ds2.py`: experimento separado para classificacao de estados relacionados a queda.
- `dash/`: interface web para upload de imagem e visualizacao das previsoes.
- `best_by_f1.keras`: checkpoint utilizado pelo dashboard.
- `Checkpoint/F1checkpoint.py`: callback customizado que salva o melhor modelo por F1 macro.
- `requirements.txt`: dependencias do projeto.

## Objetivo

O repositorio demonstra um fluxo completo de estudo em visao computacional:

- preparacao de datasets de imagem;
- treinamento com transfer learning;
- avaliacao por F1 macro;
- geracao de predicoes;
- empacotamento de um modelo treinado;
- inferencia em interface web simples.

## Bases de Dados

O projeto foi estruturado para trabalhar com duas bases distintas, ambas removidas da publicacao.

### Human Action Recognition

Consumida por `preparing_ds.py`, esperada em `Human_Action_Recognition/`:

- `Training_set.csv`: arquivo de treino com colunas `filename` e `label`;
- `Testing_set.csv`: arquivo de teste com coluna `filename`;
- `train/`: imagens de treino;
- `test/`: imagens de teste/inferencia.

As classes sao obtidas diretamente das labels presentes em `Training_set.csv`.

### Fall Dataset

Consumida por `preparing_ds2.py`, esperada em `fall_dataset/`:

- `images/train`;
- `images/val`;
- `labels/train`;
- `labels/val`.

As classes fixas no codigo sao:

- `falling_person`;
- `lying_person`;
- `standing_person`.

O material original nao registra com precisao nome oficial, link ou licenca dessas bases. Por isso, a documentacao descreve apenas estrutura e uso observados no codigo.

## Como Executar o Dashboard

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Execute:

```bash
cd dash
python app.py
```

Se `Human_Action_Recognition/Training_set.csv` nao estiver presente, o dashboard ainda funciona, mas exibe nomes genericos de classe (`class_0`, `class_1`, etc.).

## Cuidados de Publicacao

Nao foram publicados datasets brutos, ambiente virtual, arquivos de submissao ou predicoes geradas. O checkpoint `best_by_f1.keras` foi mantido porque permite testar a inferencia sem retreinar o modelo.

## Limitacoes

Este projeto e educacional. Para uso real, seria necessario documentar a origem/licenca dos dados, validar vieses, medir desempenho por classe e criar um fluxo de inferencia mais robusto.
