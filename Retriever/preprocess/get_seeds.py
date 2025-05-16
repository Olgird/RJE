import json
import os

from config import cfg


def get_seeds():
    dataset = cfg.dataset
    input_dir = cfg.processed_dataset
    output_dir = cfg.tmp_dir
    output_file = output_dir + "seeds.txt"
    if dataset == "webqsp":
        input_list = ["train_raw.json", "test_raw.json"]
    elif dataset == "cwq":
        input_list = ["train_raw.json", "dev_raw.json", "test_raw.json"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_file):
        return

    entities = set()

    for input_file in input_list:
        input_path = input_dir + input_file
        with open(input_path) as f:
            for line in f:
                q_obj = json.loads(line)
                entities.update(
                    {entity for entity in q_obj["topic_entities"] if entity}
                )

    with open(output_file, "w") as f:
        for entity in entities:
            f.write(entity + "\n")
