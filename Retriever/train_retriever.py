import logging
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config import cfg

NEG_NUM = 15


class CustomTrainingArgs:
    def __init__(
        self,
        output_dir,
        learning_rate,
        per_device_train_batch_size,
        num_train_epochs,
        save_interval,
    ):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.save_interval = save_interval


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())

    def __getitem__(self, index):
        dict = {}
        for key in self.keys:
            dict[key] = {
                "input_ids": self.data_dict[key]["input_ids"][index],
                "attention_mask": self.data_dict[key]["attention_mask"][index],
            }
        return dict

    def __len__(self):
        return len(self.data_dict[self.keys[0]]["input_ids"])


def contrasive_loss(q_emb, pos_emb, neg_emb, temperature=0.1):
    # q_emb: [batch_size, 768]
    # pos_emb: [batch_size, 768]
    # neg_emb: [batch_size, NEG_NUM, 768]
    batch_size = q_emb.shape[0]

    pos_score = F.cosine_similarity(q_emb, pos_emb, dim=-1) / temperature

    q_emb_expanded = q_emb.unsqueeze(1).expand_as(neg_emb)
    neg_score = F.cosine_similarity(q_emb_expanded, neg_emb, dim=-1) / temperature

    # Concatenate positive and negative scores [batch_size, NEG_NUM+1]
    logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
    # Create labels, where positive samples are marked as 0
    labels = torch.zeros(batch_size, dtype=torch.long).to(q_emb.device)

    loss = F.cross_entropy(logits, labels)

    # dot product
    # pos_score = torch.sum(q_emb * pos_emb, dim=-1) / temperature
    # neg_score = torch.sum(q_emb.unsqueeze(1) * neg_emb, dim=-1) / temperature

    return loss


def main():
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M")
    dataset = cfg.dataset
    if dataset == "webqsp":
        training_args = CustomTrainingArgs(
            output_dir=cfg.retriever["train"]["output_dir"] + time_string + "/",
            learning_rate=5e-5,
            per_device_train_batch_size=32,
            num_train_epochs=5,
            save_interval=500,
        )
    elif dataset == "cwq":
        training_args = CustomTrainingArgs(
            output_dir=cfg.retriever["train"]["output_dir"] + time_string + "/",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            save_interval=5000,
        )

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    path = cfg.pretrained_model["roberta_base"]
    tokenizer = AutoTokenizer.from_pretrained(path)
    train_file = cfg.retriever["train"]["input_path"]

    log_file = training_args.output_dir + "/train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="w",
    )

    df = pd.read_csv(train_file)
    df["question"] = df["question"].replace("[SEP]", tokenizer.sep_token, regex=False)
    train_data_dict = df.to_dict(orient="list")

    model = AutoModel.from_pretrained(path)

    for key in train_data_dict.keys():
        train_data_dict[key] = tokenizer(
            train_data_dict[key],
            padding="max_length",
            truncation=True,
            max_length=cfg.MAX_LEN,
            return_tensors="pt",
        )

    train_dataset = CustomDataset(train_data_dict)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)

    total_steps = training_args.num_train_epochs * len(train_dataloader)

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,  # 10% of total_steps used for warm-up
        num_training_steps=total_steps,
    )

    model.train()

    print("Start training")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Number of training steps: {total_steps}")
    print(f"Number of batches per epoch: {len(train_dataloader)}")
    progress_bar = tqdm(range(total_steps), desc="Training", disable=False)

    current_steps = 0
    for epoch in range(training_args.num_train_epochs):
        total_loss = 0
        num_batches = 0
        recent_loss = 0

        for i, batch in enumerate(train_dataloader):
            # Move data to the correct device
            # batch = {k: v.to(device) for k, v in batch.items()}
            questions = {k: v.to(device) for k, v in batch["question"].items()}
            pos = {k: v.to(device) for k, v in batch["pos"].items()}

            neg_list = []
            for index in range(NEG_NUM):
                neg = {k: v.to(device) for k, v in batch[f"neg{index}"].items()}
                neg_list.append(model(**neg, return_dict=True).pooler_output)

            """
                dim=0: the output shape is [num, batch_size, hidden_dim]. It returns num tensors of shape [batch_size, hidden_dim].
                dim=1: the output shape is [batch_size, num, hidden_dim]. It returns batch_size tensors of shape [num, hidden_dim].
            """
            neg_emb = torch.stack(neg_list, dim=1)

            q_emb = model(**questions, return_dict=True).pooler_output
            pos_emb = model(**pos, return_dict=True).pooler_output

            # Compute contrastive loss
            loss = contrasive_loss(q_emb, pos_emb, neg_emb)
            total_loss += loss.item()
            recent_loss += loss.item()
            num_batches += 1
            current_steps += 1

            if i and i % 100 == 0:
                logging.info(
                    f"Iteration {i}. Average loss for the last 100 iterations: {recent_loss / 100}"
                )
                recent_loss = 0

            if current_steps % training_args.save_interval == 0:
                model.save_pretrained(
                    training_args.output_dir + f"checkpoint-{current_steps}/"
                )

            # backpropagation
            loss.backward()
            # update parameters
            optimizer.step()
            # update learning rate
            scheduler.step()
            # clear gradients
            optimizer.zero_grad()
            # update progress bar
            progress_bar.update(1)

            del q_emb, pos_emb, neg_emb, neg_list

        avg_loss = total_loss / num_batches
        logging.info(f"Epoch {epoch+1}, Loss: {avg_loss}")
        # if epoch < training_args.num_train_epochs - 1:
        #     model.save_pretrained(training_args.output_dir + f"epoch_{epoch+1}/")
        # else:
        #     model.save_pretrained(training_args.output_dir + "final_model/")
        if epoch == training_args.num_train_epochs - 1:
            model.save_pretrained(training_args.output_dir + "final_model/")

    print("Training finished")


if __name__ == "__main__":
    main()
