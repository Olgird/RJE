class Config:
    def __init__(self, dataset):
        self.dataset = dataset
        self.processed_dataset = "../data/processed_datasets/" + dataset + "/"
        self.tmp_dir = "../data/tmp/" + dataset + "/"
        self.localgraph_dir = "../data/localgraph/" + dataset + "/"
        self.train_dir = "../data/train/" + dataset + "/"
        self.result_dir = "../data/results/" + dataset + "/"
        self.MAX_LEN = 32 if dataset == "webqsp" else 64

        

        self.pretrained_model = {"roberta_base": "FacebookAI/roberta-base"}


        self.localgraph = {
            "seed_path": self.tmp_dir + "localgraph_seed.txt",

            "path": self.localgraph_dir + "localgraph.json",
        }

        self.retriever = {
            "train": {
                "input_path": self.train_dir + "train_retriever.csv",
                "output_dir": "model_ckpt/" + dataset + "/retriever/",
            },
            "final_model": "model_ckpt/" + dataset + "/retriever/final_model/",
        }


cfg = Config("webqsp")
