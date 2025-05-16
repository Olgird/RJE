import random
import argparse
import json
import os

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

def pairwise_ranking_loss(pos_score, neg_score, margin=0.1):
    pos_score = pos_score.unsqueeze(1)
    losses = torch.clamp(neg_score - pos_score + margin, min=0)

    return losses.mean()
def read_json(file_path):

    try:
        # 打开并读取文件内容
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        print(f"成功从 {file_path} 读取数据")
        return data
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

def random_select(input_list, target_length=15):
    n = len(input_list)
    

    if n == 0:
        return []
    
    if n >= target_length:
        return random.sample(input_list, target_length)
    
    selected = input_list.copy()
    additional = random.choices(input_list, k=target_length - n)
    selected.extend(additional)
    return selected

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

class SingleTowerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("FacebookAI/roberta-base")
        self.classifier = torch.nn.Linear(768, 1) 

    def forward(self, input_ids, attention_mask):
        
        cls_embedding =  self.model(input_ids, attention_mask=attention_mask, return_dict=True).pooler_output

        logits = self.classifier(cls_embedding)
        return logits.squeeze()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--POS_NUM", type=int,
                        default=5)
    parser.add_argument("--NEG_NUM", type=int,
                        default=15)
    
    parser.add_argument("--learning_rate", type=float,
                        default=2e-5)
    
    parser.add_argument("--save_interval", type=int,
                        default=200)

    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=32)
    parser.add_argument("--num_train_epochs", type=int,
                        default=3)   

    parser.add_argument("--out_dir", type=str,
                        default="") 
    
    parser.add_argument("--margin", type=float,
                        default=0.1)
    
    parser.add_argument("--hard_pos", type=int,
                        default=1)
    
    args = parser.parse_args()


    qa_paths_list_top5 = read_json("qa_paths_list_top5_webqsp_train_data.json")

    pos_neg_dict_list = []


    uncover = 0
    for qa_paths in tqdm(qa_paths_list_top5):

        entities_rels_list = qa_paths["entities_rels_list"]
        answer_list = []
        for answer in qa_paths["answers"]:
            answer_list.append(answer["text"])

        pos_neg_dict = {"question":qa_paths["question"],"pos":[],"neg":[]}

        
        # for entities_rels in entities_rels_list:

        for i in range(len(entities_rels_list)):
            entities_rels = entities_rels_list[i]
            for e_r in entities_rels:
                for e in e_r:
                    flag_cover = False
                    for a in answer_list:
                        if a in e or e in a:
                            flag_cover = True
                            break

                if flag_cover:
                    pos_neg_dict["pos"].append(e_r)
                    if i!= 0 :
                        for _ in range(args.hard_pos):
                            pos_neg_dict["pos"].append(e_r)
                else:
                    pos_neg_dict["neg"].append(e_r)

        if len(pos_neg_dict["pos"]) == 0:
            # print(qa_paths['question'])
            uncover += 1

        elif len(pos_neg_dict["neg"]) == 0:
            # print(qa_paths['question'])
            pass
            
        else:
            pos_neg_dict_list.append(pos_neg_dict)


    path = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(path)


    train_data = []
    POS_NUM = args.POS_NUM
    NEG_NUM = args.NEG_NUM
    for pos_neg_dict in pos_neg_dict_list:

        pos_num = min(POS_NUM, len(pos_neg_dict["pos"]))

        
        random.shuffle(pos_neg_dict["pos"])
        pos_rels = pos_neg_dict["pos"][:pos_num]

        for pos_rel in pos_rels:

            train_dict = {"question": pos_neg_dict["question"],}
            train_dict["pos"] =  pos_neg_dict["question"] + tokenizer.sep_token + " → ".join(pos_rel)

            

            neg_rels = random_select(pos_neg_dict["neg"], NEG_NUM)

            for index in range(NEG_NUM):
                train_dict["neg" + str(index)] =  pos_neg_dict["question"] + tokenizer.sep_token + " → ".join(neg_rels[index])


            train_data.append(train_dict)


    learning_rate = args.learning_rate
    per_device_train_batch_size = args.per_device_train_batch_size
    num_train_epochs = args.num_train_epochs
    save_interval = args.save_interval
    MAX_LEN = 128


    stm = SingleTowerModel()

    train_data_dict = dict()

    for data in train_data:

        for key,item in data.items():

            if key not in train_data_dict.keys():

                train_data_dict[key] = []

            train_data_dict[key].append(item)


    for key in train_data_dict.keys():
            train_data_dict[key] = tokenizer(
                train_data_dict[key],
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )


    train_dataset = CustomDataset(train_data_dict)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    stm = stm.to(device)


    total_steps = num_train_epochs * len(train_dataloader)

    optimizer = AdamW(stm.parameters(), lr=learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.1,  # 10% of total_steps used for warm-up
        num_training_steps=total_steps,
    )

    stm.train()



    print("Start training")
    progress_bar = tqdm(range(total_steps), desc="Training", disable=False)

    current_steps = 0

    for epoch in range(num_train_epochs):
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
                neg_list.append(stm(**neg))

            """
                dim=0: the output shape is [num, batch_size, hidden_dim]. It returns num tensors of shape [batch_size, hidden_dim].
                dim=1: the output shape is [batch_size, num, hidden_dim]. It returns batch_size tensors of shape [num, hidden_dim].
            """
            neg_score = torch.stack(neg_list, dim=1)

            # q_emb = stm(**questions, return_dict=True)
            pos_score = stm(**pos)

            # Compute contrastive loss
            # loss = contrasive_loss_stm(pos_score, neg_score)
            loss = pairwise_ranking_loss(pos_score, neg_score, args.margin)
            
            total_loss += loss.item()
            recent_loss += loss.item()
            num_batches += 1
            current_steps += 1

            if i and i % 100 == 0:
                print(f"Iteration {i}. Average loss for the last 100 iterations: {recent_loss / 100}")
                recent_loss = 0

            if current_steps % save_interval == 0:
                checkpoint_path = os.path.join(
                        args.out_dir,
                        f"checkpoint-{current_steps}"
                    )


                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(stm.state_dict(), args.out_dir+"/" + f"checkpoint-{current_steps}/"+"single_tower_roberta.pt")


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

            del  pos_score, neg_score, neg_list

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Average loss={avg_loss}")

        if epoch == num_train_epochs - 1:
            # stm.save_pretrained("entity_path_ranker_SingleTowerModel/final_model/")
            checkpoint_path = os.path.join(
                        args.out_dir,
                        "final_model"
                    )
            os.makedirs(checkpoint_path, exist_ok=True)

            torch.save(stm.state_dict(), args.out_dir + "/" + "final_model/" + "single_tower_roberta.pt")


