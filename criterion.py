import torch
from torch import nn

def cal_sim(feat_i, feat_j, temperature):
    feat_i = feat_i / feat_i.norm(dim=len(feat_i.shape)-1, keepdim=True)
    feat_j = feat_j / feat_j.norm(dim=len(feat_j.shape)-1, keepdim=True)
    return feat_i @ feat_j.t() / temperature

class SetCriterion(nn.Module):
    def __init__(self, dataset_config, loss_weight_dict):
        super().__init__()
        self.dataset_config = dataset_config
        self.loss_weight_dict = loss_weight_dict

        self.loss_functions = {
            "loss_fmri_prediction": self.loss_fmri_prediction,
            "loss_fmri_label": self.loss_fmri_label,
            "loss_eeg_label": self.loss_eeg_label,
            "loss_fmri_eeg": self.loss_fmri_eeg,
        }


    def contrastive_loss_batch_without_break(self, objs_feats, objs_labels, bad_tr, temperature=0.1):
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")

        loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        total_loss = torch.tensor(0, dtype=float).to(device)
        valid_obj_cnt = 1
        cnt_continue = 0

        for obj_idx in range(objs_feats.shape[0]):
            if bad_tr[obj_idx] > 0:
                cnt_continue += 1
                continue
            
            obj_feature = objs_feats[obj_idx].unsqueeze(0)

            obj_label = objs_labels[obj_idx]
            neg_obj_idxs = torch.where(objs_labels != obj_label)[0]
            neg_obj_idxs = [i for i in neg_obj_idxs if bad_tr[i] == 0]
           
            if len(neg_obj_idxs) > 0:
                neg_objs = objs_feats[neg_obj_idxs, :]
                neg_loss = cal_sim(obj_feature, neg_objs, temperature)
            else:
                continue

            pos_objs_idxs = torch.where(objs_labels == obj_label)[0]
            pos_objs_idxs = [i for i in pos_objs_idxs if i != obj_idx and bad_tr[i] == 0]

            if len(pos_objs_idxs) > 0:
                pos_objs = objs_feats[pos_objs_idxs, :]
                pos_loss = cal_sim(obj_feature, pos_objs, temperature).t()
            else:
                pos_loss = torch.full((1, 1), 1/temperature)
                valid_obj_cnt -= 1
            pos_loss = pos_loss.to(device)
            neg_loss = neg_loss.to(device)
            logits = torch.cat([pos_loss, neg_loss.repeat(pos_loss.shape[0],1)],dim=1).to(device)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
            
            curr_loss = loss_criterion(logits, labels)
            total_loss += curr_loss
            valid_obj_cnt += 1
        
        total_loss /= valid_obj_cnt
        return total_loss
    
    def contrastive_loss(self, objs_feats, objs_labels, bad_tr, temperature=0.1):
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")

        loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        total_loss = torch.tensor(0, dtype=float).to(device)
        valid_obj_cnt = 1
        cnt_continue = 0

        for obj_idx in range(objs_feats.shape[0]):
            if bad_tr[obj_idx] > 0:
                cnt_continue += 1
                continue
            
            obj_feature = objs_feats[obj_idx].unsqueeze(0)

            obj_label = objs_labels[obj_idx]
            neg_obj_idxs = torch.where(objs_labels != obj_label)[0]
            neg_obj_idxs = [i for i in neg_obj_idxs if bad_tr[i] == 0]
           
            if len(neg_obj_idxs) > 0:
                neg_objs = objs_feats[neg_obj_idxs, :]
                neg_loss = cal_sim(obj_feature, neg_objs, temperature)
            else:
                continue

            pos_objs_idxs = torch.where(objs_labels == obj_label)[0]
            pos_objs_idxs = [i for i in pos_objs_idxs if i != obj_idx and bad_tr[i] == 0]

            if len(pos_objs_idxs) > 0:
                pos_objs = objs_feats[pos_objs_idxs, :]
                pos_loss = cal_sim(obj_feature, pos_objs, temperature).t()
            else:
                pos_loss = torch.full((1, 1), 1/temperature)
                valid_obj_cnt -= 1
            pos_loss = pos_loss.to(device)
            neg_loss = neg_loss.to(device)
            logits = torch.cat([pos_loss, neg_loss.repeat(pos_loss.shape[0],1)],dim=1).to(device)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
            
            curr_loss = loss_criterion(logits, labels)
            total_loss += curr_loss
            valid_obj_cnt += 1
            break
        
        total_loss /= valid_obj_cnt
        return total_loss
        
    def loss_fmri_prediction(self, outputs, targets):
        loss_prediction_criterion = nn.CrossEntropyLoss()
        y_true = outputs["eeg_index"].long()
        y_pred = outputs["predictions"]["vigilance_head_logits"]
        bad_tr = outputs["bad_tr"].reshape(-1)
        mask = bad_tr == 0
        y_pred_filtered = y_pred[mask]
        y_pred_filtered = y_pred_filtered.reshape(y_pred_filtered.shape[0], -1)
        y_true_filtered = y_true[mask].reshape(-1)
        curr_loss = loss_prediction_criterion(y_pred_filtered, y_true_filtered)
        return {"loss_fmri_prediction": curr_loss}

    def loss_fmri_label(self, outputs, targets):
        shape = outputs["eeg_index"].shape
        bad_tr = outputs["bad_tr"].reshape(shape[0]*shape[1])
        fmri_objs_labels = outputs["eeg_index"].reshape(shape[0]*shape[1], 1)
        fmri_objs_feats = outputs["fmri_feats"]
        curr_loss = self.contrastive_loss(fmri_objs_feats, fmri_objs_labels, bad_tr)
        return {"loss_fmri_label": curr_loss}
    
    def loss_eeg_label(self, outputs, targets): 
        shape = outputs["eeg_index"].shape
        bad_tr = outputs["bad_tr"].reshape(shape[0]*shape[1])
        eeg_objs_labels = outputs["eeg_index"].reshape(shape[0]*shape[1], 1)
        eeg_objs_feats = outputs["eeg_feats"]
        curr_loss = self.contrastive_loss(eeg_objs_feats, eeg_objs_labels, bad_tr)
        return {"loss_eeg_label": curr_loss}

    def loss_fmri_eeg(self, outputs, targets):
        fmri_objs_map_feats = outputs["fmri_map_feats"]
        eeg_objs_map_feats = outputs["eeg_map_feats"] 
        total_objs_map_feats = torch.cat([fmri_objs_map_feats, eeg_objs_map_feats], dim=0)

        shape = outputs["eeg_index"].shape
        total_objs_labels_part = outputs["eeg_index"].reshape(shape[0]*shape[1], 1)
        total_objs_labels = torch.cat([total_objs_labels_part, total_objs_labels_part], dim=0)

        bad_tr_part = outputs["bad_tr"].reshape(shape[0]*shape[1])
        bad_tr = torch.cat([bad_tr_part, bad_tr_part], dim=0)
        curr_loss_00 = self.contrastive_loss(total_objs_map_feats, total_objs_labels, bad_tr)
        curr_loss_01 = self.contrastive_loss(fmri_objs_map_feats, total_objs_labels_part, bad_tr_part)
        curr_loss_02 = self.contrastive_loss(eeg_objs_map_feats, total_objs_labels_part, bad_tr_part)
        curr_loss = curr_loss_00 + curr_loss_01 + curr_loss_02
        return {"loss_fmri_eeg": curr_loss}
    
    def single_output_forward(self, outputs, targets):
        losses = {}
        for f in self.loss_functions: 
            loss_wt_key = f + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                curr_loss = self.loss_functions[f](outputs, targets)
                losses.update(curr_loss)

        final_loss = 0.0
        for w in self.loss_weight_dict:
            if self.loss_weight_dict[w] > 0:
                losses[w.replace("_weight", "")] *= self.loss_weight_dict[w]
                final_loss += losses[w.replace("_weight", "")]
        return final_loss, losses
    
    def forward(self, outputs, targets):
        loss, loss_dict = self.single_output_forward(outputs, targets)
        return loss, loss_dict


def build_criterion(args, dataset_config):
    loss_weight_dict = {
        "loss_fmri_prediction_weight": args.loss_fmri_prediction_weight,
        "loss_fmri_label_weight": args.loss_fmri_label_weight,
        "loss_eeg_label_weight": args.loss_eeg_label_weight,
        "loss_fmri_eeg_weight": args.loss_fmri_eeg_weight,
    }
    criterion = SetCriterion(dataset_config, loss_weight_dict)
    return criterion