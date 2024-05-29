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
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler 
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


def run():
    args=prase_args()
    config = get_config(args.config_file)
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    data_dist = torch.load(config["common"]["data_file"])
    #X_train, X_test, y_train, y_test = train_test_split(data_dist['data'], data_dist['target'], test_size=0.1, random_state=seed)
    X_train, X_test, y_train, y_test = data_dist['data'][:9000],data_dist['data'][9000:],data_dist['target'][:9000],data_dist['target'][9000:]

    train_dataset = MyDatasets(X_train, y_train)
    train_sampler = DistributedSampler(train_dataset,shuffle=True)
    test_dataset = MyDatasets(X_test, y_test)
    model = model_builder(config,device)
    tokenizer = tokenizer_builder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], eps = 1e-8, weight_decay=config["train"]["weight_decay"])
    train_iter = DataLoader(train_dataset,batch_size=config["train"]["batch_size"],sampler=train_sampler)
    test_iter = DataLoader(test_dataset,batch_size=1)
    loss_function = nn.CrossEntropyLoss()
    ckpt_file = build_log(model,config)
    Train(config, model, tokenizer,config["train"]["epoch_num"], train_iter, test_iter, loss_function, optimizer,device,ckpt_file)
if __name__ =='__main__':
    run()