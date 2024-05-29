import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def count_correct(y_h, y):
    y_h = torch.argmax(y_h.cpu(), dim=1)
    return sum(y_h == y)


def calculate_metrics(y_label, y_pre):
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_label, y_pre).ravel()
    
    # 计算准确率、精确率、召回率和F1分数
    accuracy = accuracy_score(y_label, y_pre)
    precision = precision_score(y_label, y_pre)
    recall = recall_score(y_label, y_pre)
    f1 = f1_score(y_label, y_pre)
    
    # 将结果整理成字典返回
    results = {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "score": (tp+fp)/(tp+fp+fn+tn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return results
