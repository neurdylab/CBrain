# implemented following 3DETR's eval_det.py
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix

def eval_f1_wrapper(arguments):
    pred, gt = arguments
    f1 = eval_f1_cls(pred, gt)
    precision = eval_precision_cls(pred, gt)
    re = eval_recall_cls(pred, gt)
    return f1, precision, re 

def eval_f1_cls(
        pred, gt
):
    f1 = f1_score(gt, pred, average=None)
    return f1

def eval_precision_cls(
        pred, gt
):
    precision = precision_score(gt, pred, average=None)
    return precision

def eval_recall_cls(
        pred, gt
):
    re = recall_score(gt, pred, average=None)
    return re

def eval_accuracy_cls(
        pred, gt
):
    classes = sorted(np.unique(gt))
    cm = confusion_matrix(gt, pred)
    ac = []
    for i, classname in enumerate(classes):
        print(classname)
        print(i)
        print("individual num:")
        print(cm[i, i])
        print("total_num:")
        print(cm[i, :].sum())
        if cm[i, :].sum() == 0:
            ac.append(0)
            print(0)
            continue
        class_accuracy = cm[i, i] / cm[i, :].sum()
        print(class_accuracy)
        ac.append(class_accuracy)
    return ac


def eval_f1_multiprocessing(
        pred_all, gt_all, bad_tr_all
):
    pred = []
    gt = []

    total_class = {}
    for img_id in pred_all.keys():
        for pred_idx in range(len(pred_all[img_id])):
            if bad_tr_all[img_id][pred_idx] > 0:
                continue
            pred_classname = pred_all[img_id][pred_idx]
            classname = round(pred_classname)
            pred.append(classname)
            if classname not in total_class:
                total_class[classname] = 1
    for img_id in gt_all.keys():
        for gt_idx in range(len(gt_all[img_id])):
            if bad_tr_all[img_id][gt_idx] > 0:
                continue
            gt_classname = gt_all[img_id][gt_idx]
            classname = round(gt_classname)
            gt.append(classname)
            if classname not in total_class:
                total_class[classname] = 1

    f1 = {}
    precision = {}
    re = {}
    report = classification_report(gt, pred, output_dict=True)
    print(report)
    classnames = [0, 1]
    for i in classnames:
        if str(i) in report.keys():
            report_data = report[str(i)]
            f1[i] = report_data['f1-score']
            precision[i] = report_data['precision']
            re[i] = report_data['recall']
        else:
            f1[i] = 0.0
            precision[i] = 0.0
            re[i] = 0.0

    return f1, precision, re