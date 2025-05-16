
class Config:

    def __init__(self,dataset):
        
        self.dataset = dataset 
        self.processed_dataset = "data/processed_datasets/" + dataset + "/"

        self.result_dir = "data/results/" + dataset + "/" + "train/"
        

        self.train_dir = "data/train/" + dataset + "/"
        self.pretrained_model = {"roberta_base": "FacebookAI/roberta-base"}
        self.MAX_LEN = 128
        if dataset == "webqsp":
            
            self.retriever = {
                "train": {
                    "input_path": self.train_dir + "train_retriever.csv",
                    "output_dir": "model_ckpt/" + dataset + "/retriever/",
                },
                "final_model": "model_ckpt/webqsp/retriever/final_model",
            }

        elif dataset == "cwq":
            self.retriever = {
                "train": {
                    "input_path": self.train_dir + "train_retriever.csv",
                    "output_dir": "model_ckpt/" + dataset + "/retriever/",
                },
                "final_model": "model_ckpt/cwq/retriever/final_model",
            }

        self.inference = {

            "input_path": self.processed_dataset + "train.json",
            "output_dir": self.result_dir,
        }


cfg = Config("cwq")
    