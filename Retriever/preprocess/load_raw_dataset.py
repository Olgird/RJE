import json
import os

from config import cfg
from knowledge_graph.knowledge_graph_freebase import KnowledgeGraphFreebase


def is_ent(tp_str):
    if len(tp_str) < 3:
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        return True
    return False


def find_entity(sparql_str):
    str_lines = sparql_str.split("\n")
    # tp_list = re.finditer(r"ns:[a-z]\..* ", str_lines)
    ent_set = set()

    for line in str_lines[1:]:
        if "ns:" not in line:
            continue

        spline = line.strip().split()
        for item in spline:
            str = item
            if str.startswith("("):
                str = str.lstrip("(")
            if not str.startswith("ns:"):
                continue
            str = str[3:].replace("(", "")
            str = str.replace(")", "")
            if is_ent(str):
                if str.endswith("?x"):
                    str = str[:-2]
                ent_set.add(str)
    return ent_set


def load_webqsp(input_path) -> list:
    with open(input_path) as f:
        dataset = json.load(f)
    dataset = dataset["Questions"]

    data_list = []
    for json_obj in dataset:
        id = json_obj["QuestionId"]
        question = json_obj["ProcessedQuestion"]
        for parse in json_obj["Parses"]:
            topic_entities = [parse["TopicEntityMid"]]
            topic_entity_names = [parse["TopicEntityName"]]

            answer_list = []
            for answer_obj in parse["Answers"]:
                if answer_obj["AnswerType"] == "Entity":
                    new_obj = {}
                    new_obj["kb_id"] = answer_obj["AnswerArgument"]
                    new_obj["text"] = answer_obj["EntityName"]
                    answer_list.append(new_obj)
            if not answer_list:
                continue
            data_list.append(
                {
                    "id": id,
                    "question": question,
                    "topic_entities": topic_entities,
                    "topic_entity_names": topic_entity_names,
                    "answers": answer_list,
                }
            )
    return data_list


def load_cwq(input_path, name) -> list:
    with open(input_path) as f:
        if name == "ComplexWebQuestions_test_wans.json":
            dataset = [json.loads(line) for line in f]
        else:
            dataset = json.load(f)

    kg = KnowledgeGraphFreebase()

    data_list = []
    for json_obj in dataset:
        id = json_obj["ID"]
        question = json_obj["question"]

        if name == "ComplexWebQuestions_test_wans.json":
            answer_list = json_obj["answers"]
        else:
            answer_list = []
            for answer_obj in json_obj["answers"]:
                new_obj = {}
                new_obj["kb_id"] = answer_obj["answer_id"].lstrip(":")
                new_obj["text"] = answer_obj["answer"]
                answer_list.append(new_obj)

        sparql_str = json_obj["sparql"]
        topic_entities = list(find_entity(sparql_str))
        topic_entity_names = [kg.get_ent_name(ent) for ent in topic_entities]

        data_list.append(
            {
                "id": id,
                "question": question,
                "topic_entities": topic_entities,
                "topic_entity_names": topic_entity_names,
                "answers": answer_list,
            }
        )

    return data_list


def load_raw_dataset() -> None:
    dataset = cfg.dataset
    output_dir = cfg.processed_dataset

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.listdir(output_dir):
        return

    if dataset == "webqsp":
        input_list = ["WebQSP.train.json", "WebQSP.test.json"]
        output_list = ["train.json", "test.json"]
        dataset_dir = "data/datasets/WebQSP/data/"
        data_file_list = [load_webqsp(dataset_dir + name) for name in input_list]
    elif dataset == "cwq":
        input_list = [
            "ComplexWebQuestions_train.json",
            "ComplexWebQuestions_dev.json",
            "ComplexWebQuestions_test_wans.json",
        ]
        output_list = ["train_raw.json", "dev_raw.json", "test_raw.json"]
        dataset_dir = "data/datasets/ComplexWebQuestions/"
        data_file_list = [load_cwq(dataset_dir + name, name) for name in input_list]

    else:
        raise NotImplementedError

    for data_file, output_name in zip(data_file_list, output_list):
        output_path = output_dir + output_name
        with open(output_path, "w") as f:
            for item in data_file:
                f.write(json.dumps(item) + "\n")
