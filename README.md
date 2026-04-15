# cv-activity-recognition-dashboard

Projeto de visão computacional para classificação de atividades humanas e estudo de detecção de quedas, com treinamento em TensorFlow e dashboard Dash para inferência por imagem.

## Conteúdo

- `preparing_ds.py`: pipeline de treinamento para classificação multiclasse de ações humanas com EfficientNetV2B0.
- `preparing_ds2.py`: experimento separado para classificação de estados relacionados a queda.
- `dash/`: interface web para upload de imagem e visualização das previsões.
- `best_by_f1.keras`: checkpoint utilizado pelo dashboard.
- `Checkpoint/F1checkpoint.py`: callback customizado que salva o melhor modelo por F1 macro.
- `requirements.txt`: dependências do projeto.

## Objetivo

O repositório demonstra um fluxo completo de estudo em visão computacional:

- preparação de datasets de imagem;
- treinamento com transfer learning;
- avaliação por F1 macro;
- geração de predições;
- empacotamento de um modelo treinado;
- inferência em interface web simples.

## Bases de Dados

O projeto foi estruturado para trabalhar com duas bases distintas, ambas removidas da publicação.

### Human Action Recognition

Consumida por `preparing_ds.py`, esperada em `Human_Action_Recognition/`:

- `Training_set.csv`: arquivo de treino com colunas `filename` e `label`;
- `Testing_set.csv`: arquivo de teste com coluna `filename`;
- `train/`: imagens de treino;
- `test/`: imagens de teste/inferência.

As classes são obtidas diretamente das labels presentes em `Training_set.csv`.

### Fall Dataset

Consumida por `preparing_ds2.py`, esperada em `fall_dataset/`:

- `images/train`;
- `images/val`;
- `labels/train`;
- `labels/val`.

As classes fixas no código são:

- `falling_person`;
- `lying_person`;
- `standing_person`.

O material original não registra com precisão nome oficial, link ou licença dessas bases. Por isso, a documentação descreve apenas estrutura e uso observados no código.

## Como Executar o Dashboard

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute:

```bash
cd dash
python app.py
```

Se `Human_Action_Recognition/Training_set.csv` não estiver presente, o dashboard ainda funciona, mas exibe nomes genéricos de classe (`class_0`, `class_1`, etc.).

## Cuidados de Publicação

Não foram publicados datasets brutos, ambiente virtual, arquivos de submissão ou predições geradas. O checkpoint `best_by_f1.keras` foi mantido porque permite testar a inferência sem retreinar o modelo.

## Limitações

Este projeto é educacional. Para uso real, seria necessário documentar a origem/licença dos dados, validar vieses, medir desempenho por classe e criar um fluxo de inferência mais robusto.
