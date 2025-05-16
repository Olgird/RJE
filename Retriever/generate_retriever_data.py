import json
import time
from multiprocessing import Manager, Pool

from tqdm import tqdm

from preprocess.search_path_score import (
    search_path_score_multiprocess,
    search_path_score_wrapper,
)
from retriever.get_retriever_train_data import get_retriever_train_data

if __name__ == "__main__":

    def search_path_and_score():
        output_path, num_processes, total_len, tasks = search_path_score_multiprocess()
        if not tasks:
            return
        path_score_list = []
        with Manager() as manager:
            counter = manager.Value("i", 0)
            counter_lock = manager.Lock()
            with Pool(processes=num_processes) as pool:
                pbar = tqdm(total=total_len)
                results = pool.imap(
                    search_path_score_wrapper,
                    [
                        (
                            task[0],
                            task[1],
                            task[2],
                            counter,
                            counter_lock,
                        )
                        for task in tasks
                    ],
                )

                while True:
                    pbar.n = counter.value
                    pbar.refresh()
                    if pbar.n >= total_len:
                        pbar.close()
                        break
                    time.sleep(1)

                for result in results:
                    for line in result:
                        path_score_list.append(line)

                with open(output_path, "a") as outf:
                    for dict in path_score_list:
                        outf.write(json.dumps(dict, ensure_ascii=False) + "\n")

    search_path_and_score()
    get_retriever_train_data()
