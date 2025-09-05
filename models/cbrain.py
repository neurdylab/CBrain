import torch
import torch.nn as nn
from models.transformer import (TransformerEncoder,
                                TransformerEncoderLayer)

from models.helpers import GenericMLP
from models.helpers import (WEIGHT_INIT_DICT)


class CBrain(nn.Module):
    def __init__(
        self,
        fmri_spatial_encoder,
        fmri_temporal_encoder,
        fmri_mapping_layer,
        eeg_pre_encoder,
        eeg_sp_encoder,
        eeg_tp_encoder,
        eeg_mapping_layer,
        predictors,
        dataset_config,
    ):
        super().__init__()
        self.fmri_spatial_encoder = fmri_spatial_encoder
        self.fmri_temporal_encoder = fmri_temporal_encoder
        self.fmri_mapping_layer = fmri_mapping_layer
        self.eeg_pre_encoder = eeg_pre_encoder
        self.eeg_sp_encoder = eeg_sp_encoder
        self.eeg_tp_encoder = eeg_tp_encoder
        self.eeg_mapping_layer = eeg_mapping_layer
        self.predictors = predictors
        self.dataset_config = dataset_config
        self._reset_parameters()
    
    def _reset_parameters(self):
        func = WEIGHT_INIT_DICT["xavier_uniform"]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)
    
    def run_fmri_encoders_average(self, fmri_raw):
        _, fmri_sp_feats, _ = self.fmri_spatial_encoder(fmri_raw.transpose(1, 2))
        _, fmri_te_feats, _ = self.fmri_temporal_encoder(fmri_raw)
        fmri_feats = (fmri_sp_feats.transpose(1, 2) + fmri_te_feats) / 2
        return fmri_feats
    
    def run_fmri_mapping(self, fmri_feats):
        fmri_map_feats = self.fmri_mapping_layer(fmri_feats)
        return fmri_map_feats

    def run_eeg_preencoder(self, eeg_original_feats):
        downsample_eeg = self.eeg_pre_encoder(eeg_original_feats)
        return downsample_eeg
    
    def run_eeg_encoder(self, eeg_pre_feats):
        _, eeg_feats, _  = self.eeg_encoder(eeg_pre_feats)
        return eeg_feats
    
    def run_eeg_encoders_average(self, eeg_pre_feats):
        _, eeg_sp_feats, _ = self.eeg_sp_encoder(eeg_pre_feats.transpose(1, 2))
        _, eeg_te_feats, _ = self.eeg_tp_encoder(eeg_pre_feats)
        eeg_feats = (eeg_sp_feats.transpose(1, 2) + eeg_te_feats) / 2
        return eeg_feats
    
    def run_eeg_encoders_multiplication(self, eeg_pre_feats):
        _, eeg_sp_feats, _ = self.eeg_sp_encoder(eeg_pre_feats.transpose(1, 2))
        _, eeg_te_feats, _ = self.eeg_tp_encoder(eeg_pre_feats)
        temporal_weights = torch.matmul(eeg_te_feats.transpose(1, 2), eeg_te_feats) 
        temporal_weights = temporal_weights / torch.norm(temporal_weights, p='fro')
        eeg_feats = torch.bmm(eeg_sp_feats.transpose(1, 2), temporal_weights) 
        return eeg_feats
    
    def run_eeg_mapping(self, eeg_feats):
        eeg_map_feats = self.eeg_mapping_layer(eeg_feats)
        return eeg_map_feats
    
    def get_predictions(self, fmri_feats):
        fmri_feed_feats = fmri_feats.reshape(fmri_feats.shape[0], -1)
        vigilance = self.predictors["vigilance_head"](fmri_feed_feats)
        predicted_label = torch.argmax(vigilance, dim=1) 
        predicted_name = predicted_label
        vigilance_result = vigilance.reshape((-1, 1, 2))
        predicted_name_result = predicted_name.reshape(-1, 1)
        predictions = {
            "vigilance_head_logits": vigilance_result,
            "vigilance_head": predicted_name_result,
        }
        return predictions
    

    def forward(self, inputs, is_train=True):
        fmri_raw = inputs["fmri"] 
        eeg_raw = inputs["eeg"] 
        input_bad_tr = inputs["bad_tr"] 
        input_eeg_index = inputs["eeg_index"] 

        downsample_eeg = self.run_eeg_preencoder(eeg_raw) 
        fmri_feats = self.run_fmri_encoders_average(fmri_raw) 
        predictions = self.get_predictions(fmri_feats)

        eeg_feats = self.run_eeg_encoders_average(downsample_eeg)
        fmri_map_feats = self.run_fmri_mapping(fmri_feats.transpose(1, 2)).transpose(1, 2)
        eeg_map_feats = self.run_eeg_mapping(eeg_feats.transpose(1, 2)).transpose(1, 2)

        ret_dict = {}
        ret_dict["predictions"] = predictions
        ret_dict["bad_tr"] = input_bad_tr
        ret_dict["eeg_index"] = input_eeg_index
        ret_dict["eeg_feats"] = eeg_feats.reshape(eeg_feats.shape[0], -1)
        ret_dict["fmri_feats"] = fmri_feats.reshape(fmri_feats.shape[0], -1)
        ret_dict["eeg_map_feats"] = eeg_map_feats.reshape(eeg_map_feats.shape[0], -1)
        ret_dict["fmri_map_feats"] = fmri_map_feats.reshape(fmri_map_feats.shape[0], -1)
        return ret_dict


def build_fmri_sp_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        d_model=args.fmri_sp_enc_dim,
        nhead=args.fmri_sp_enc_nhead,
        dim_feedforward=args.fmri_sp_enc_ffn_dim,
        dropout=args.fmri_sp_enc_dropout,
        activation=args.fmri_sp_enc_activation,
    )
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=args.fmri_sp_enc_nlayers
    )
    return encoder


def build_fmri_te_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        d_model=args.fmri_te_enc_dim,
        nhead=args.fmri_te_enc_nhead,
        dim_feedforward=args.fmri_te_enc_ffn_dim,
        dropout=args.fmri_te_enc_dropout,
        activation=args.fmri_te_enc_activation,
    )
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=args.fmri_te_enc_nlayers
    )
    return encoder


def build_fmri_mapping(args):
    c_in = 66
    mapping_layer = nn.Sequential(
        nn.Linear(c_in, 32, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(32, 10, bias=False),
        nn.ReLU(inplace=True)
    )
    return mapping_layer


def build_eeg_preencoder(args):
    embedding = nn.Conv1d(in_channels=26, out_channels=26, kernel_size=525, stride=525, padding=0)
    return embedding


def build_eeg_sp_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        d_model=args.eeg_sp_enc_dim,
        nhead=args.eeg_sp_enc_nhead,
        dim_feedforward=args.eeg_sp_enc_ffn_dim,
        dropout=args.eeg_sp_enc_dropout,
        activation=args.eeg_sp_enc_activation,
    )
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=args.eeg_sp_enc_nlayers
    )
    return encoder


def build_eeg_tp_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        d_model=args.eeg_enc_dim,
        nhead=args.eeg_enc_nhead,
        dim_feedforward=args.eeg_enc_ffn_dim,
        dropout=args.eeg_enc_dropout,
        activation=args.eeg_enc_activation,
    )
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=args.eeg_enc_nlayers
    )
    return encoder


def build_eeg_mapping(args):
    c_in = 26
    mapping_layer = nn.Sequential(
        nn.Linear(c_in, 20, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(20, 10, bias=False),
        nn.ReLU(inplace=True)
    )
    return mapping_layer 


def build_predictors(args, dataset_config):
    vigilance_cls_head = GenericMLP(
            input_dim=10*66,
            hidden_dims=[64, 32, 16],
            output_dim=2,
            norm_fn_name="bn1d",
            activation="leakyrelu",
            use_conv=False,
            dropout=0.2,
            hidden_use_bias=True,
        )
    predictors = [
        ("vigilance_head", vigilance_cls_head),
    ]
    return nn.ModuleDict(predictors)


def build_cbrain(args, dataset_config):
    fmri_spatial_encoder = build_fmri_sp_encoder(args)
    fmri_temporal_encoder = build_fmri_te_encoder(args)
    fmri_mapping_layer = build_fmri_mapping(args)
    eeg_pre_encoder = build_eeg_preencoder(args)
    eeg_sp_encoder = build_eeg_sp_encoder(args)
    eeg_tp_encoder = build_eeg_tp_encoder(args)
    eeg_mapping_layer = build_eeg_mapping(args)
    predictors = build_predictors(args, dataset_config)

    model = CBrain(
        fmri_spatial_encoder,
        fmri_temporal_encoder,
        fmri_mapping_layer,
        eeg_pre_encoder,
        eeg_sp_encoder,
        eeg_tp_encoder,
        eeg_mapping_layer,
        predictors,
        dataset_config,
    )
    return model