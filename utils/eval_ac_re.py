# implemented following 3DETR's eval_det.py
import numpy as np
from multiprocessing import Pool

def eval_ac_re_cls_wrapper(arguments):
    pred_cls, gt_cls, samples_num, sample_length, classname = arguments
    ac, re, cls_name, tp_cnt, len_pred_cls, len_gt_cls, samples_num, tn_cnt = eval_ac_re_cls(pred_cls, gt_cls, samples_num, sample_length, classname)
    return (ac, re, cls_name, tp_cnt, len_pred_cls, len_gt_cls, samples_num, tn_cnt)

def eval_ac_re_cls(
        pred_cls_raw, gt_cls_raw, samples_num, sample_length, classname
):
    pred_cls = np.array([x * sample_length + i
                        for x in pred_cls_raw.keys()
                        for i in pred_cls_raw[x]])
    gt_cls = np.array([x * sample_length + i
                      for x in gt_cls_raw.keys()
                      for i in gt_cls_raw[x]])

    common_elements = np.intersect1d(pred_cls, gt_cls)

    tp_cnt = len(common_elements)
    tn_cnt = samples_num - len(pred_cls) - len(gt_cls) + tp_cnt
    ac = (tp_cnt + tn_cnt) / samples_num

    if len(gt_cls) == 0:
        re = 0.0
    else:
        re = tp_cnt / len(gt_cls)
    
    return ac, re, classname, tp_cnt, len(pred_cls), len(gt_cls), samples_num, tn_cnt

def eval_ac_re_multiprocessing(
        pred_all, gt_all, bad_tr_all
):
    pred = {}
    gt = {}
    samples_num = 0
    sample_length = 0

    
    for img_id in pred_all.keys(): 
        samples_num += (len(pred_all[img_id]) - sum(np.array(bad_tr_all[img_id])))
        if sample_length == 0:
            sample_length = len(pred_all[img_id])
        for pred_idx in range(len(pred_all[img_id])):
            # bad tr, remove
            if bad_tr_all[img_id][pred_idx] > 0:
                continue
            pred_classname = pred_all[img_id][pred_idx]
            classname = round(pred_classname)

            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((pred_idx))

    for img_id in gt_all.keys():
        for gt_idx in range(len(gt_all[img_id])):
            if bad_tr_all[img_id][gt_idx] > 0:
                continue
            gt_classname = gt_all[img_id][gt_idx]
            classname = round(gt_classname)
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append((gt_idx))  

    ac = {}
    re = {}

    p = Pool(processes=10)
    ret_values = p.map(
        eval_ac_re_cls_wrapper,
        [
            (pred[classname], gt[classname], samples_num, sample_length, classname)
            for classname in gt.keys()
            if classname in pred
        ],
    )
    p.close()
   

    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            ac[classname], re[classname], cls_name, tp_cnt, len_pred_cls, len_gt_cls, samples_num, tn_cnt = ret_values[i]
            print("Classname: " + str(cls_name))
            print(f"tp_cnt = len(common_elements): {tp_cnt}")
            print(f"len(pred_cls): {len_pred_cls}")
            print(f"len(gt_cls): {len_gt_cls}")
            print(f"samples_num: {samples_num}")
            print(f"tn_cnt = samples_num - len(pred_cls) - len(gt_cls) + tp_cnt: {tn_cnt}")
            print(f"ac = (tp_cnt + tn_cnt) / samples_num: {ac[classname]}")
            print(f"re = tp_cnt / len(pred_cls): {re[classname]}")
            print("\n")
        else:
            ac[classname] = 0
            re[classname] = 0
    return ac, re