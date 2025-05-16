import json
import os

from config import cfg

banned_entities = {
    "m.05zppz",  # Male
    "m.02zsn",  # Female
    "m.04ztj",  # Marriage
    "m.01mp",  # Country
    "m.01y2hnl",  # College/University
    "m.019v9k",  # Bachelor's degree
    "m.01y2hn6",  # School
    "m.02nsjvf",  # Voice
    "m.01nt",  # Region
    "m.0jsg2m",  # Film
    "m.0qcr0",  # Cancer
    "m.01xryvm",  # Book
    "m.02_7k44",  # Dated
    "m.01xpjyz",  # Airport
    "m.0290ngj",  # Vocals
    "m.0_sv_90",  # Lead Vocals
    "m.02_99rl",  # Engaged to
    "m.0bpgx",  # High School
    "m.01xs05k",  # River
    "m.07jdr",  # Train
    "m.0j749",  # Motto
    "m.0342h",  # Guitar
    "m.029j_",  # DVD
    "m.01jfsb",  # Thriller
    "m.018jz",  # Baseball
    "m.018w8",  # Basketball
    "m.048zv9l",  # Senator
    "m.060c4",  # President
    "m.0fkvn",  # Governor
    "m.01wb7",  # Church
    "m.02822",  # Drama
    "m.07c52",  # Television
    "m.01mh",  # Continent
    "m.025t3bg",  # Air travel
    "m.014cnc",  # Student
    "m.01tf_6",  # Drug overdose
    "m.02j8z",  # Evolution
    "m.01zdtb",  # Acting President
    "m.01gkgk",  # Member of Congress
    "m.0173tc",  # Chapel
    "m.02h76fz",  # Military Conflict
    "m.0fj9f",  # Politician
    "m.02xlf",  # Fiction
}


def remove_banned_entities(input_path, entity_name_dict):
    with open(input_path) as f:
        data_list = [json.loads(line) for line in f]
    for data in data_list:
        topic_entities = data["topic_entities"]
        topic_entity_names = data["topic_entity_names"]
        filtered = [
            (e, n)
            for e, n in zip(topic_entities, topic_entity_names)
            if e not in banned_entities
        ]
        if filtered:
            data["topic_entities"], data["topic_entity_names"] = list(zip(*filtered))
        else:
            print("No topic entity: " + data["question"])
            data["topic_entities"], data["topic_entity_names"] = [], []

        for name in data["topic_entity_names"]:
            if name in entity_name_dict:
                entity_name_dict[name] += 1
            else:
                entity_name_dict[name] = 1
    return data_list


def filter_entities():
    dataset = cfg.dataset
    data_dir = cfg.processed_dataset
    if dataset == "webqsp":
        return
    elif dataset == "cwq":
        entity_name_dict = {}
        input_list = ["train_raw.json", "dev_raw.json", "test_raw.json"]
        output_list = ["train.json", "dev.json", "test.json"]

        if os.path.exists(data_dir + "names.txt"):
            return

        data_file_list = [
            remove_banned_entities(data_dir + name, entity_name_dict)
            for name in input_list
        ]
        for data_file, output_name in zip(data_file_list, output_list):
            output_path = data_dir + output_name
            with open(output_path, "w") as f:
                for item in data_file:
                    f.write(json.dumps(item) + "\n")

        sorted_list = sorted(entity_name_dict.items(), key=lambda x: x[1], reverse=True)

        with open(data_dir + "names.txt", "w") as f:
            for tuple in sorted_list:
                f.write(f"{tuple[0]}\t{tuple[1]}\n")
