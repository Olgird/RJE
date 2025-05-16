# RJE

![image](assets/RJE.png)

## Knowledge Graph

Before running the system, you need to set up **Freebase** locally. Please follow the instructions in the [Freebase-Setup guide](https://github.com/GasolSun36/ToG/tree/main/Freebase).

## Installation

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

## Running (Example: CWQ Dataset)

### 1. Train the Relation Retriever

Run the following script to train the relation retriever:

```bash
bash ./Retriever/train_retriever.sh
```

### 2. Train the Inference Path Ranker

Run the following script to train the path ranker:

```bash
bash train_cwq_ranker.sh
```

### 3. Prepare for Running

Prepare necessary files with:

```bash
python prepare_running.py
```

### 4. Run the Model

To execute the main pipeline, run:

```bash
python rje.py \
  --dataset "cwq" \
  --max_length 2048 \
  --select_num 30 \
  --path_num 10 \
  --LLM_type "gpt35" 
```


## Evaluation

```bash
cd eval
python eval.py
```