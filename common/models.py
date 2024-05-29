import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification,GPT2Config,GPT2ForSequenceClassification

def tokenizer_builder(config):
    tokenizer=AutoTokenizer.from_pretrained(config["model"]["model_hf"],cache_dir="./cache")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
def model_builder(config,device):
    model=AutoModelForSequenceClassification.from_pretrained(config["model"]["model_hf"], 
    num_labels=2,problem_type="multi_label_classification",cache_dir="./cache",ignore_mismatched_sizes=True).to(device)
    model.config.pad_token_id = model.config.eos_token_id
    model=torch.nn.parallel.DistributedDataParallel(model)
    return model
