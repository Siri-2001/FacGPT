import torch,json
from torch.utils.data import Dataset

class MyDatasets(Dataset):
    def __init__(self, x,y):
        super().__init__()
        self.data = x
        self.target = y
        assert all([i==1 or i==0 for i in y])

    def __len__(self):
        assert len(self.data) == len(self.target)
        return len(self.data)
    def __getitem__(self, item):
        return "Question:{} ;Golden answer:{} ;AI-generate answer:{} ;Judge:".format(self.data[item]["question"],self.data[item]["golden_answer"],str(self.data[item]["answer"])), self.target[item]

class InferDatasets(Dataset):
    def __init__(self, config):
        super().__init__()
        self.json = json.load(open(config["infer"]["infer_json"],'r',encoding='utf-8'))
        self.engines = config["infer"]["engines"]
        self.data = [
        {"question": each["question"],
         "golden_answer": each["golden_answer"],
         "answer": each[f"answer_{engine}"]} 
         for each in self.json for engine in self.engines if not each['improper']]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return "Question:{} ;Golden answer:{} ;AI-generate answer:{} ;Judge:".format(self.data[item]["question"],self.data[item]["golden_answer"],str(self.data[item]["answer"]))
def build_dataset():
    ''' 生成训练集和数据集，待确定'''
    return
