#!/bin/bash

python preprocess_dataset.py

python run_get_localgraph.py

python generate_retriever_data.py

python train_retriever.py
