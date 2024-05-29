import random, os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 确定有关GPU使用的相关问题
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
from common.config import get_config
from common.arguments import *
from common.models import *
from common.Datasets import *
from common.Trainer import *
from common.Logging import *
from torch.utils.data.dataloader import DataLoader
import numpy as np




# 固定随机种子
seed = 1024
random.seed(seed)     # python的随机性
np.random.seed(seed)  # np的随机性
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)            # torch的CPU随机性，为CPU设置随机种子
torch.cuda.manual_seed(seed)       # torch的GPU随机性，为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # torch的GPU随机性，为所有GPU设置随机种子
# 固定随机种子


def infer():
    args=prase_args()
    config = get_config(args.config_file)
    os.makedirs(config["infer"]["infer_result_dir"], exist_ok=True)
    device = torch.device(config["infer"]["device"])
    model = GPT2ForSequenceClassification.from_pretrained(config["infer"]["checkpoint"]).to(device)
    tokenizer = tokenizer_builder(config)
    infer_total_sample = 0
    start_time = time.time()
    y_pres=[]
    infer_dataset = InferDatasets(config)
    infer_iter=DataLoader(infer_dataset,batch_size=1)
    with open(os.path.join(config["infer"]["infer_result_dir"],config["infer"]["infer_result_file"]),'w') as fout:
        print(config["infer"]["infer_result_dir"],file=fout)
        with torch.no_grad():
            model.eval()
            for idx, x in enumerate(tqdm.tqdm(infer_iter)):
                print(x)
                inputs = tokenizer(x, padding=True, truncation=True, return_tensors="pt").to(device)
                output = model(**inputs)
                logits = output.logits
                y_pre = torch.argmax(logits.cpu(), dim=1)
                y_pres.append(y_pre)
                print(y_pre.item())
                print(y_pre.item(),file=fout)
                infer_total_sample += len(logits)
    execution_time = time.time() - start_time
    print("execution time per case:{}".format(execution_time/len(y_pres)))

if __name__ =='__main__':
    infer()