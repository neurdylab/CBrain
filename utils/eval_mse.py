# implemented following 3DETR's eval_det.py
from sklearn.metrics import mean_squared_error, mean_absolute_error

def eval_mse_wrapper(arguments):
    pred, gt = arguments
    mae, mse = eval_mse(pred, gt)
    return mae, mse

def eval_mse(
        pred, gt
):
    mae = mean_absolute_error(pred, gt)
    mse = mean_squared_error(pred, gt)
    return mae, mse

def eval_mse_multiprocessing(
        pred_all, gt_all
):
    pred = []
    gt = []
    for img_id in pred_all.keys():
        pred += pred_all[img_id]
    for img_id in gt_all.keys():
        gt += gt_all[img_id]

    mae, mse = eval_mse_wrapper((pred, gt))
    return mae, mse