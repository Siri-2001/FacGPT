import os,tqdm
from common.metrics import *
from common.Logging import Log
import datetime, time
def evaluate(config, model, tokenizer,test_iter, device):
    eval_total_loss = 0
    eval_total_sample = 0
    start_time = time.time()
    y_pres=[]
    y_labels=[]
    with torch.no_grad():
        model.eval()
        for idx,(x, y) in enumerate(test_iter):
            y_labels.append(y)
            inputs = tokenizer(x, padding=True, truncation=True, return_tensors="pt").to(device)
            labels = torch.nn.functional.one_hot(torch.tensor(y).clone().detach(), num_classes=2).to(torch.float)
            output = model(**inputs, labels=labels.to(device))
            logits = output.logits
            loss = output.loss
            y_pre = torch.argmax(logits.cpu(), dim=1)
            y_pres.append(y_pre)
            eval_total_loss += loss.item()
            eval_total_sample += len(logits)
    metrics=calculate_metrics(y_labels, y_pres)
    execution_time = time.time() - start_time
    return eval_total_loss/eval_total_sample,metrics,execution_time/len(y_labels)


def Train(config, model, tokenizer, epoch_num, train_iter, test_iter, loss_function, optimizer, device,ckpt_file):
    model = model.to(device)

    for i in tqdm.tqdm(range(1, epoch_num+1)):
        train_total_loss = 0
        train_sample_num = 0
        train_total_correctnum = 0
        epoch_start_time = time.time()
        for x, y in tqdm.tqdm(train_iter):
            model.train()
            optimizer.zero_grad()
            inputs = tokenizer(x, padding=True, truncation=True,max_length=768, return_tensors="pt").to(device)
            labels = torch.nn.functional.one_hot(torch.tensor(y).clone().detach(), num_classes=2).to(torch.float)
            output = model(**inputs, labels=labels.to(device))
            logits = output.logits
            loss = output.loss
            loss.backward()
            train_total_correctnum += count_correct(logits, y)
            train_total_loss += loss.item()
            train_sample_num += len(logits)
            optimizer.step()
        avg_train_loss = train_total_loss/train_sample_num
        train_acc = train_total_correctnum/train_sample_num
        avg_eval_loss, eval_metircs, exetime_percase = evaluate(config,model, tokenizer,test_iter, device)
        epoch_end_time = time.time()
        Log('epoch %s: (time: %s) Train Loss:%s Eval Loss:%s '
            'Train Acc:%.2f Eval Metrics:%s  Evaluatetime_percase:%f' % (str(i),
                                               str(epoch_end_time-epoch_start_time),
                                               avg_train_loss, avg_eval_loss,
                                               train_acc, str(eval_metircs),exetime_percase),
                                               os.path.join(config["train"]["checkpoint_path"],
                                                            ckpt_file, 'log'+'Epoch{}_F1{}'.format(i,str(eval_metircs["f1_score"]).replace('.','_')))).log()
        if i % config["train"]["sckp_epoch"] == 0:
            model.module.save_pretrained(os.path.join(config["train"]["checkpoint_path"], ckpt_file,'Epoch{}'.format(i)))
    return







