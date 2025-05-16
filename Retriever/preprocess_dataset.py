from preprocess.filter_entities import filter_entities
from preprocess.get_notional_words import get_notional_words
from preprocess.load_raw_dataset import load_raw_dataset


def run():
    load_raw_dataset()
    filter_entities()
    get_notional_words()


if __name__ == "__main__":
    run()
