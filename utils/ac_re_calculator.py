# implemented following ap_calculator in 3DETR
import numpy as np
from collections import OrderedDict
from utils.eval_ac_re import eval_ac_re_multiprocessing
from utils.eval_f1 import eval_f1_multiprocessing
from utils.eval_correlation import eval_corr_multiprocessing
from utils.eval_mse import eval_mse_multiprocessing

def get_a_config_dict(
    dataset_config,
):
    config_dict = {
        "dataset_config": dataset_config,
    }
    return config_dict

class ACRECalculator(object):
    def __init__(
            self,
            dataset_config,
            a_config_dict=None,
    ):
        if a_config_dict is None:
            a_config_dict = get_a_config_dict(
                dataset_config=dataset_config
            )
        self.a_config_dict = a_config_dict
        self.reset()
    
    def make_pred_list(self, predictions):
        batch_pred = []
        bsize = predictions.shape[0]
        for i in range(bsize):
            batch_pred.append(
                [
                    (predictions[i, j].item())
                    for j in range(predictions.shape[1])
                ]
            )
        return batch_pred
    
    def make_gt_list(self, gt_predictions):
        batch_gt = []
        bsize = gt_predictions.shape[0]
        for i in range(bsize):
            batch_gt.append(
                [
                    (gt_predictions[i, j].item())
                    for j in range(gt_predictions.shape[1])
                ]
            )
        return batch_gt
    
    def make_bad_tr_list(self, gt_bad_tr):
        batch_bad_tr = []
        bsize = gt_bad_tr.shape[0]
        for i in range(bsize):
            batch_bad_tr.append(
                [
                    (gt_bad_tr[i, j].item())
                    for j in range(gt_bad_tr.shape[1])
                ]
            )
        return batch_bad_tr
    
    def step_meter(self, outputs, targets):
        if "predictions" in outputs:
            outputs = outputs["predictions"]
        self.step(
            predictions=outputs,
            gt_predictions=targets,
        )
    
    def step(self, predictions, gt_predictions):
        batch_pred = self.make_pred_list(predictions["vigilance_head"])
        batch_gt = self.make_gt_list(gt_predictions["eeg_index"])
        batch_bad_tr = self.make_bad_tr_list(gt_predictions["bad_tr"])
        self.accumulate(batch_pred, batch_gt, batch_bad_tr)
    
    def accumulate(self, batch_pred, batch_gt, batch_bad_tr):
        bsize = len(batch_pred)
        assert bsize == len(batch_gt)
        for i in range(bsize):
            self.pred[self.scan_cnt] = batch_pred[i]
            self.gt[self.scan_cnt] = batch_gt[i]
            self.bad_tr[self.scan_cnt] = batch_bad_tr[i]
            self.scan_cnt += 1
    
    def compute_metrics(self):
        ret_dict = OrderedDict()

        if len(self.pred) == 0 and len(self.gt) == 0:
            print("no prev step meter result")
            print("get to normal multi processing code")
        
        ac, re = eval_ac_re_multiprocessing(self.pred, self.gt, self.bad_tr)
        ac_vals = []
        for key in sorted(ac.keys()):
            clsname = str(key)
            ret_dict["%s Accuracy" % (clsname)] = ac[key]
            ac_vals.append(ac[key])
        ac_vals = np.array(ac_vals, dtype=np.float32)
        ret_dict["mAC"] = ac_vals.mean()

        re_vals = []
        for key in sorted(ac.keys()):
            clsname = str(key)
            try:
                ret_dict["%s Recall" % (clsname)] = re[key]
                re_vals.append(re[key])
            except:
                ret_dict["%s Recall" % (clsname)] = 0
                re_vals.append(0)
        re_vals = np.array(re_vals, dtype=np.float32)
        ret_dict["mAR"] = re_vals.mean()

        f1, precision, new_re = eval_f1_multiprocessing(self.pred, self.gt, self.bad_tr)
        for key in sorted(f1.keys()):
            clsname = str(key)
            ret_dict["%s F1 Score" % (clsname)] = f1[key]
        f1_vals = np.array(list(f1.values()), dtype=np.float32)
        ret_dict["mF1"] = f1_vals.mean()

        for key in sorted(precision.keys()):
            clsname = str(key)
            ret_dict["%s Precision" % (clsname)] = precision[key]
        precision_vals = np.array(list(precision.values()), dtype=np.float32)
        ret_dict["mPrecision"] = precision_vals.mean()

        for key in sorted(new_re.keys()):
            clsname = str(key)
            ret_dict["%s New_re" % (clsname)] = new_re[key]
        new_re_vals = np.array(list(new_re.values()), dtype=np.float32)
        ret_dict["mNew_re"] = new_re_vals.mean()

        corr, p = eval_corr_multiprocessing(self.pred, self.gt)
        ret_dict["Corr"] = corr
        ret_dict["P-value"] = p 

        mae, mse = eval_mse_multiprocessing(self.pred, self.gt)
        ret_dict["MAE"] = mae
        ret_dict["MSE"] = mse

        return ret_dict
    
    def __str__(self):
        overall_ret = self.compute_metrics()
        return self.metrics_to_str(overall_ret)
    
    def metrics_to_str(self, overall_ret, per_class=True):
        mAC_strs = []
        mAR_strs = []
        mF1_strs = []

        mPrecision_strs = []
        mNew_re_strs = []

        mCorr_strs = []
        mPVal_strs = []
        mMAE_strs = []
        mMSE_strs = []
        per_class_metrics = []

        mAC = overall_ret["mAC"] * 100
        mAC_strs.append(f"{mAC:.2f}")
        mAR = overall_ret["mAR"] * 100
        mAR_strs.append(f"{mAR:.2f}")
        mF1 = overall_ret["mF1"] * 100
        mF1_strs.append(f"{mF1:.2f}")

        mPrecision = overall_ret["mPrecision"] * 100
        mPrecision_strs.append(f"{mPrecision:.2f}")
        mNew_re = overall_ret["mNew_re"] * 100
        mNew_re_strs.append(f"{mNew_re:.2f}")
        
        mCorr = overall_ret["Corr"] 
        mCorr_strs.append(f"{mCorr:.2f}")
        mPVal = overall_ret["P-value"] 
        mPVal_strs.append(f"{mPVal:.2f}")

        mMAE = overall_ret["MAE"]
        mMAE_strs.append(f"{mMAE:.2f}")
        mMSE = overall_ret["MSE"]
        mMSE_strs.append(f"{mMSE:.2f}")
        
        overall_list = ["mAC", "mAR", "mF1", "mPrecision", "mNew_re", "mNew_ac", "Corr", "P-value", "MAE", "MSE"]
        if per_class:
            per_class_metrics.append("-" * 5)
            for x in list(overall_ret.keys()):
                if x in overall_list:
                    pass
                else:
                    met_str = f"{x}: {overall_ret[x]*100:.2f}"
                    per_class_metrics.append(met_str)

        
        ac_header = ["mAC"]
        ac_str = ", ".join(ac_header)
        ac_str += ": " + ", ".join(mAC_strs)
        ac_str += "\n"

        ar_header = ["mAR"]
        ac_str += ", ".join(ar_header)
        ac_str += ": " + ", ".join(mAR_strs)
        ac_str += "\n"

        f1_header = ["mF1"]
        ac_str += ", ".join(f1_header)
        ac_str += ": " + ", ".join(mF1_strs)
        ac_str += "\n"

        precision_header = ["mPrecision"]
        ac_str += ", ".join(precision_header)
        ac_str += ": " + ", ".join(mPrecision_strs)
        ac_str += "\n"
        
        new_re_header = ["mNew_re"]
        ac_str += ", ".join(new_re_header)
        ac_str += ": " + ", ".join(mNew_re_strs)
        ac_str += "\n"
        
        corr_header = ["Corr"]
        ac_str += ", ".join(corr_header)
        ac_str += ": " + ", ".join(mCorr_strs)
        ac_str += "\n"

        p_header = ["P-value"]
        ac_str += ", ".join(p_header)
        ac_str += ": " + ", ".join(mPVal_strs)
        ac_str += "\n"

        mae_header = ["MAE"]
        ac_str += ", ".join(mae_header)
        ac_str += ": " + ", ".join(mMAE_strs)
        ac_str += "\n"

        mse_header = ["MSE"]
        ac_str += ", ".join(mse_header)
        ac_str += ": " + ", ".join(mMSE_strs)

        if per_class:
            per_class_metrics = "\n".join(per_class_metrics)
            ac_str += "\n"
            ac_str += per_class_metrics
        
        return ac_str
    
    def metrics_to_dict(self, overall_ret):
        metrics_dict = {}
        metrics_dict["mAC"] = overall_ret["mAC"] * 100
        metrics_dict["mAR"] = overall_ret["mAR"] * 100
        metrics_dict["mF1"] = overall_ret["mF1"] * 100
        metrics_dict["mPrecision"] = overall_ret["mPrecision"] * 100
        metrics_dict["mNew_re"] = overall_ret["mNew_re"] * 100
        metrics_dict["Corr"] = overall_ret["Corr"]
        metrics_dict["P-value"] = overall_ret["P-value"]
        metrics_dict["MAE"] = overall_ret["MAE"]
        metrics_dict["MSE"] = overall_ret["MSE"]
        return metrics_dict
    
    def reset(self):
        self.pred = {}
        self.gt = {}
        self.bad_tr = {}
        self.scan_cnt = 0