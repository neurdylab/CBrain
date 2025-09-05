# implemented following 3DETR's eval_det.py
from scipy.stats import pearsonr

def eval_corr_wrapper(arguments):
    pred, gt = arguments
    corr, p = eval_corr(pred, gt)
    return corr, p

def eval_corr(
        pred, gt
):
    corr, p = pearsonr(pred, gt)
    return corr, p

def eval_corr_multiprocessing(
        pred_all, gt_all
):
    pred = []
    gt = []
    for img_id in pred_all.keys():
        pred += pred_all[img_id]
    for img_id in gt_all.keys():
        gt += gt_all[img_id]
    corr, p = eval_corr_wrapper((pred, gt))
    return corr, p