import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import joblib

class SVMHierarchy(nn.Module):
    def __init__(
        self,
        args,
        dataset_config,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.best_rbf_params = None
        self.best_poly_params = None
        self.best_rbf_score_acc1 = 0
        self.best_rbf_score_acc2 = 0
        self.best_poly_score_acc1 = 0
        self.best_poly_score_acc2 = 0
        self.layer = nn.Linear(10, 5)

    def get_predictions_rbf(self, fmri_feat):
        vigilance = torch.tensor(self.model_rbf.predict(fmri_feat))
        logits = self.model_poly.decision_function(fmri_feat)
        scaler = MinMaxScaler(feature_range=(0, 1))
        logits_2d = np.zeros((2, len(logits)))
        logits_2d[0] = np.where(logits < 0, -logits, 0)
        logits_2d[1] = np.where(logits > 0, logits, 0)
        logits_scaled = scaler.fit_transform(logits_2d.T).T 
        logits_scaled = torch.tensor(logits_scaled).reshape((fmri_feat.shape[0], 2))
        predictions = {
            "vigilance_head": vigilance.reshape((fmri_feat.shape[0], 1)),
            "vigilance_head_logits": logits_scaled,
        }
        return predictions
    
    def get_predictions_poly(self, fmri_feat):
        vigilance = torch.tensor(self.model_poly.predict(fmri_feat))
        logits = self.model_poly.decision_function(fmri_feat)
        scaler = MinMaxScaler(feature_range=(0, 1))
        logits_2d = np.zeros((2, len(logits)))
        logits_2d[0] = np.where(logits < 0, -logits, 0)
        logits_2d[1] = np.where(logits > 0, logits, 0)
        logits_scaled = scaler.fit_transform(logits_2d.T).T 
        logits_scaled = torch.tensor(logits_scaled).reshape((fmri_feat.shape[0], 2))
        predictions = {
            "vigilance_head": vigilance.reshape(fmri_feat.shape[0], 1),
            "vigilance_head_logits": logits_scaled,
        }
        return predictions

    def forward(self, inputs, is_train=True):
        fmri_raw = inputs["fmri"].reshape(inputs["fmri"].shape[0], -1)

        if is_train:
            C_values = [2**i for i in range(-1, 26)]
            gamma_values = [2**i for i in range(-1, 26)]
            degree_values = [2, 3, 4, 5]  # Polynomial degrees
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            eeg_index = inputs["eeg_index"].ravel()
            
            print("begin gamma search")
            for gamma in gamma_values:
                for C in C_values:
                    acc1, acc2 = [], []
                    print(f"gamma: {gamma}" + f" C: {C}")
                    for train_idx, test_idx in kf.split(fmri_raw):
                        X_train, X_test = np.array(fmri_raw[train_idx]), np.array(fmri_raw[test_idx])
                        y_train, y_test = np.array(eeg_index[train_idx]), np.array(eeg_index[test_idx])
                        temp_model = SVC(kernel='rbf', C=C, gamma=gamma)
                        temp_model.fit(X_train, y_train)
                        y_pred = temp_model.predict(X_test)
                        acc1.append(accuracy_score(y_test.ravel(), y_pred.ravel()))
                        acc_class = []
                        for cls_name in np.unique(eeg_index):
                            cls_idx = y_test == cls_name 
                            acc_class.append(accuracy_score(y_test[cls_idx].ravel(), y_pred[cls_idx].ravel()))
                        acc2.append(np.mean(acc_class))
                    mean_acc1 = np.mean(acc1)
                    mean_acc2 = np.mean(acc2)
                    if mean_acc1 > self.best_rbf_score_acc1:
                        # if mean_acc2 > self.best_rbf_score_acc2:
                        self.best_rbf_score_acc1 = mean_acc1
                        self.best_rbf_score_acc2 = mean_acc2
                        self.best_rbf_params = {'C': C, 'gamma': gamma}

            
            print("begin degree search")
            for degree in degree_values:
                for C in C_values:
                    print(f"degree: {degree}" + f" C: {C}")
                    acc1, acc2 = [], []
                    for train_idx, test_idx in kf.split(fmri_raw):
                        X_train, X_test = np.array(fmri_raw[train_idx]), np.array(fmri_raw[test_idx])
                        y_train, y_test = np.array(eeg_index[train_idx]), np.array(eeg_index[test_idx])
                        temp_model = SVC(kernel='poly', C=C, degree=degree)
                        temp_model.fit(X_train, y_train)
                        y_pred = temp_model.predict(X_test)
                        acc1.append(accuracy_score(y_test.ravel(), y_pred.ravel()))
                        acc_class = []
                        for cls_name in np.unique(eeg_index):
                            cls_idx = y_test == cls_name
                            acc_class.append(accuracy_score(y_test[cls_idx].ravel(), y_pred[cls_idx].ravel()))
                        acc2.append(np.mean(acc_class))
                    mean_acc1 = np.mean(acc1)
                    mean_acc2 = np.mean(acc2)
                    if mean_acc1 > self.best_poly_score_acc1:
                        # if mean_acc2 > self.best_poly_score_acc2:
                        print(f"mean_acc1: {mean_acc1}")
                        self.best_poly_score_acc1 = mean_acc1
                        self.best_poly_score_acc2 = mean_acc2
                        self.best_poly_params = {'C': C, 'degree': degree}
            
            self.model_rbf = SVC(kernel='rbf', **self.best_rbf_params)
            self.model_poly = SVC(kernel='poly', **self.best_poly_params)
            self.model_rbf.fit(fmri_raw, eeg_index)
            self.model_poly.fit(fmri_raw, eeg_index)

            joblib.dump(self.model_rbf, "outputs/ablation_svm_hierarchy_000/model_rbf.pkl")
            joblib.dump(self.model_poly, "outputs/ablation_svm_hierarchy_000/model_poly.pkl")
            predictions = self.get_predictions_rbf(fmri_raw)
            degree = self.model_poly.degree
            gamma = self.model_poly._gamma 
            coef0 = self.model_poly.coef0
            # fmri_poly_feats = polynomial_kernel(fmri_raw, degree=degree, gamma=gamma, coef0=coef0)
            fmri_poly_feats = rbf_kernel(fmri_raw, gamma) 
        else:
            self.model_rbf = joblib.load("outputs/ablation_svm_hierarchy_000/model_rbf.pkl")
            self.model_poly = joblib.load("outputs/ablation_svm_hierarchy_000/model_poly.pkl")
            predictions = self.get_predictions_rbf(fmri_raw)
            degree = self.model_poly.degree
            gamma = self.model_poly._gamma 
            coef0 = self.model_poly.coef0
            fmri_poly_feats = rbf_kernel(fmri_raw, gamma=gamma) 
        print(f"fmri_poly_feats: {fmri_poly_feats.shape}")
        print(inputs["bad_tr"].shape)
        print(inputs["eeg_index"].shape)

        ret_dict = {}
        ret_dict["predictions"] = predictions
        ret_dict["bad_tr"] = torch.tensor(inputs["bad_tr"])
        ret_dict["eeg_index"] = torch.tensor(inputs["eeg_index"])
        ret_dict["eeg_feats"] = torch.tensor(fmri_poly_feats.reshape(fmri_poly_feats.shape[0], -1))
        ret_dict["fmri_feats"] = torch.tensor(fmri_poly_feats.reshape(fmri_poly_feats.shape[0], -1))
        ret_dict["eeg_map_feats"] = torch.tensor(fmri_poly_feats.reshape(fmri_poly_feats.shape[0], -1))
        ret_dict["fmri_map_feats"] = torch.tensor(fmri_poly_feats.reshape(fmri_poly_feats.shape[0], -1))
        return ret_dict

def build_svm_hierarchy(args, dataset_config):
    model = SVMHierarchy(args, dataset_config)
    return model