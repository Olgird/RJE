import json
import random

import nltk
from flair.data import Sentence
from flair.models import SequenceTagger

from config import cfg

NOTIONAL_WORD_NUM = None

grammar1 = """
    NP: {<RBS|RBR>?<JJ|JJR|JJS>*<NN|NNS|NNP|NNPS>+}
    VP: {<VB|VBD|VBG|VBN|VBP|VBZ>+}
"""
grammar2 = """
    NP: {<NP><IN><DT>?<NP>}
    NP: {<NP><POS><NP>}
"""
grammar3 = """
    VNP: {<VP><IN><DT>?<NP>}
    VNP: {<VP><DT>?<NP>}
    VP: {<VP><IN>}
    VP: {<VP><DT>?<RBS|RBR>}
"""

excluded_verbs = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
}

filter_str = ',.:!?")(][}{\\'


def find_subsequence(s_list, texts):
    positions = []

    for i in range(len(texts)):
        if not texts[i] == s_list[0]:
            continue
        j = 0
        k = 0
        while j < len(s_list) and i + j + k < len(texts):
            if texts[i + j + k] == s_list[j]:
                j += 1
            elif texts[i + j + k] in filter_str:
                k += 1
            else:
                break
        if j == len(s_list):
            positions.extend(range(i, i + j + k))

        # if texts[i : i + len_s] == s_list:
        #     positions.extend(range(i, i + len_s))
    return positions


def filter_topic_entity(question, texts, topic_entity_names):
    positions = []
    entity_split = dict()
    texts = [text.strip(filter_str) for text in texts]

    for entity_name in topic_entity_names:
        entity_split[entity_name] = entity_name.split()
        entity_split[entity_name] = [
            ent.strip(filter_str) for ent in entity_split[entity_name]
        ]
        # Note that the way texts and entity_name are separated is different
        positions += find_subsequence(entity_split[entity_name], texts)

    if not positions:
        for i in range(len(texts)):
            for entity_name in topic_entity_names:
                if texts[i] in entity_split[entity_name]:
                    positions.append(i)

    for position in positions:
        question[position].add_label("filter", True)


def pos_and_chunk(data_list, tagger):
    questions = [Sentence(d["question"]) for d in data_list]
    tagger.predict(questions, mini_batch_size=256)

    total_phrase_list = []

    parser1 = nltk.RegexpParser(grammar1)
    parser2 = nltk.RegexpParser(grammar2)
    parser3 = nltk.RegexpParser(grammar3)

    for question, data in zip(questions, data_list):
        topic_entity_names = [
            name.lower()
            for name in data["topic_entity_names"]
            # if name and name.lower() not in reserved_names
        ]

        texts = [token.text.lower() for token in question]

        filter_topic_entity(question, texts, topic_entity_names)

        # Preprocess the sentence, replace the excluded words with a special label
        tagged_sentence = [
            (
                token.text,
                "EXCLUDED"
                if token.text.lower() in excluded_verbs
                or token.get_labels("filter")
                or (token.text == "'s" and token.get_label("pos").value == "VBZ")
                else token.get_label("pos").value,
            )
            for token in question
        ]

        # Chunk using nltk
        chunked_1 = parser1.parse(tagged_sentence)
        chunked_2 = parser2.parse(chunked_1)
        chunked_tree = parser3.parse(chunked_2)

        chunk_types = {"VNP", "NP", "VP"}

        # Create an empty dictionary to store phrases
        phrases = {chunk_type: [] for chunk_type in chunk_types}

        # Traverse the top level of the chunked tree
        for subtree in chunked_tree:
            # If the label of this subtree is in chunk_types, extract the words in this subtree
            if type(subtree) == nltk.tree.Tree and subtree.label() in chunk_types:
                phrase = " ".join(word for word, pos in subtree.leaves())
                phrases[subtree.label()].append(phrase)

        phrase_list = phrases["VNP"] + phrases["NP"] + phrases["VP"]
        phrase_list = phrase_list[:NOTIONAL_WORD_NUM]

        if len(phrase_list) < NOTIONAL_WORD_NUM:
            first_token = texts[0]
            if first_token == "where":
                phrase_list.append("location")
            # elif first_token == "when":
            #     phrase_list.append("time")
            elif first_token == "who":
                phrase_list.append("person")

        while len(phrase_list) < NOTIONAL_WORD_NUM and len(phrase_list) > 0:
            phrase_list.append(random.choice(phrase_list))
        total_phrase_list.append(phrase_list)

    return total_phrase_list


def load_file(input_path):
    with open(input_path) as f:
        data_list = [json.loads(line) for line in f]
    return data_list


def get_notional_words():
    global NOTIONAL_WORD_NUM
    dataset = cfg.dataset
    data_dir = cfg.processed_dataset
    if dataset == "webqsp":
        NOTIONAL_WORD_NUM = 2
        file_list = ["train.json", "test.json"]
    elif dataset == "cwq":
        NOTIONAL_WORD_NUM = 3
        file_list = ["train.json", "dev.json", "test.json"]
    else:
        raise NotImplementedError

    tagger = SequenceTagger.load("flair/pos-english")

    data_file_list = [load_file(data_dir + name) for name in file_list]

    for data_file in data_file_list:
        notional_words_list = pos_and_chunk(data_file, tagger)
        for dict_, value in zip(data_file, notional_words_list):
            dict_["notional_word"] = value
    for data_file, output_name in zip(data_file_list, file_list):
        output_path = data_dir + output_name
        with open(output_path, "w") as f:
            for item in data_file:
                f.write(json.dumps(item) + "\n")
