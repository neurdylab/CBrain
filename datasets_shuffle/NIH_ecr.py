import mne
import scipy.io
import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

EEG_BASE_DIR = ""
EEG_MAT_DIR = ""
FMRI_BASE_DIR = ""
EEG_INDEX_BASE_DIR = ""
EEG_SOURCE_FORMAT = ''
FMRI_SOURCE_FORMAT = ''
EEG_INDEX_SOURCE_FORMAT = ''
EEG_FMRI_EVENT = ''
FMRI_PROC_SCANS = []
FMRI_PROC_SCANS_TRAIN = []
FMRI_PROC_SCANS_TEST = []
SCAN_TASK_DICT = {}

class NIHECRDatasetConfig(object):
    def __init__(self):
        self.default_value = 1
        self.num_classes = 2
        self.name = "NIH_ecr"

class NIHECRDataset(Dataset):
    def __init__(
            self,
            dataset_config,
            split_set="train",
            eeg_base_dir=None,
            fmri_base_dir=None,
            eeg_index_base_dir=None,
            eeg_source_format=None,
            fmri_source_format=None,
            eeg_fmri_event=None,
    ):
    
        assert split_set in ["train", "test", "val", "zero_shot", "trainval"]

        if eeg_base_dir is None:
            eeg_base_dir=EEG_BASE_DIR
        if fmri_base_dir is None:
            fmri_base_dir=FMRI_BASE_DIR
        if eeg_index_base_dir is None:
            eeg_index_base_dir=EEG_INDEX_BASE_DIR
        if eeg_source_format is None:
            eeg_source_format = EEG_SOURCE_FORMAT
        if fmri_source_format is None:
            fmri_source_format = FMRI_SOURCE_FORMAT
        if eeg_fmri_event is None:
            eeg_fmri_event = EEG_FMRI_EVENT

        self_eeg_base_dir = eeg_base_dir
        self_eeg_mat_dir = EEG_MAT_DIR
        self_fmri_base_dir = fmri_base_dir
        self_eeg_index_base_dir = eeg_index_base_dir
        self_eeg_fmri_event = eeg_fmri_event
        self_scan_task_dict = SCAN_TASK_DICT

        all_scan_names = FMRI_PROC_SCANS
        train_scan_names = FMRI_PROC_SCANS_TRAIN
        test_scan_names = FMRI_PROC_SCANS_TEST
        
        all_subject_names = {}
        for scan_name in all_scan_names:
            subject_name = scan_name[:8]
            if subject_name not in all_subject_names:
                all_subject_names[subject_name] = []
            all_subject_names[subject_name].append(scan_name)

        train_subject_names = {}
        for scan_name in train_scan_names:
            subject_name = scan_name[:8]
            if subject_name not in train_subject_names:
                train_subject_names[subject_name] = []
            train_subject_names[subject_name].append(scan_name)
        
        test_subject_names = {}
        for scan_name in test_scan_names:
            subject_name = scan_name[:8]
            if subject_name not in test_subject_names:
                test_subject_names[subject_name] = []
            test_subject_names[subject_name].append(scan_name)
        
        if split_set == "train":
            self_scan_names = train_scan_names
        elif split_set == "val":
            self_scan_names = train_scan_names
        elif split_set == "trainval":
            self_scan_names = train_scan_names
        elif split_set == "test":
            self_scan_names = test_scan_names
        elif split_set == "zero_shot":
            self_scan_names = all_scan_names
        
        result_scan_names = {}
        for scan_name in self_scan_names:
            subject_name = scan_name[:6]
            if subject_name in result_scan_names:
                result_scan_names[subject_name] += 1
            else:
                result_scan_names[subject_name] = 1
        
        self.scan_names = self_scan_names 

        bad_tr_total = []
        fmri_data_total = []
        eeg_data_total = []
        gt_total = []
        for idx in range(len(self_scan_names)):
            scan_name = self_scan_names[idx]
            eeg_scan_paths = glob.glob(os.path.join(self_eeg_base_dir, scan_name+'*.set'))
            eeg_scan_name = scan_name
            if len(eeg_scan_paths) == 0:
                scan_task = scan_name[17:20]
                if scan_task in self_scan_task_dict:
                    eeg_scan_name = scan_name[:17] + self_scan_task_dict[scan_task] + scan_name[20:]
                    eeg_scan_paths = ... # TODO: scan path
                if len(eeg_scan_paths) == 0:
                    eeg_scan_name = scan_name[:20] + '_' + scan_name[21:]
                    eeg_scan_paths = ... # TODO: scan path
            eeg_scan_path = eeg_scan_paths[0]
            fmri_scan_path = ... # TODO: scan path
            scan_vigall_index = scan_name[0:12] + "{:04d}".format(int(eeg_scan_name[12:16]) + 1)
            eeg_index_path = glob.glob(os.path.join(self_eeg_index_base_dir, scan_vigall_index+'*.mat'))
            drop_list = ['time', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6']
            zero_pad_list = ['FPz', 'POz', 'FT9', 'FT10']
            change_name_dict = {"TP9": "TP9'",
                                "TP10": "TP10'"}
            eeg_data = mne.io.read_raw_eeglab(eeg_scan_path)
            df_eeg_data = eeg_data.to_data_frame()
            df_eeg_data = df_eeg_data.rename(columns=change_name_dict)
            eeg_events, eeg_events_id = mne.events_from_annotations(eeg_data)
            eeg_fmri_code = eeg_events_id[self_eeg_fmri_event]
            eeg_fmri_event = eeg_events[eeg_events[:, 2] == eeg_fmri_code]
            fmri_raw_data = pd.read_csv(fmri_scan_path)
            fmri_data_00 = fmri_raw_data.drop(columns={"Unnamed: 0"})

            for drop_column in drop_list:
                if drop_column in df_eeg_data.columns:
                    df_eeg_data = df_eeg_data.drop([drop_column], axis=1)
            for zero_pad_column in zero_pad_list:
                if zero_pad_column not in df_eeg_data.columns:
                    df_eeg_data[zero_pad_column] = 0.0
            
            slide_cnt = (fmri_data_00.shape[0] - 2) // 5
            fmri_data = fmri_data_00.iloc[2:slide_cnt*5+2, :66]
            first_seven_threshold = eeg_fmri_event[(7)*30][0]
            df_eeg_data = df_eeg_data.iloc[first_seven_threshold:eeg_fmri_event[30*(slide_cnt*5)][0]+first_seven_threshold, :]

            eeg_index_data_raw = scipy.io.loadmat(eeg_index_path[0])
            eeg_index_data = eeg_index_data_raw['VIG_SIG'][0][0][3][2:slide_cnt*5+2]
            eeg_index_data = np.array(eeg_index_data) 
            window_size = 10
            step_size = 5
            windows = np.lib.stride_tricks.sliding_window_view(eeg_index_data, window_shape=(window_size, 1))[::step_size, :, :]
            input_eeg_index = windows.reshape(windows.shape[0], window_size)
            binary_sums = input_eeg_index.sum(axis=1)
            clusters = (binary_sums > -5).astype(int)
            eeg_mat_path = glob.glob(os.path.join(self_eeg_mat_dir, scan_name+"*.mat"))
            eeg_mat = scipy.io.loadmat(eeg_mat_path[0])
            eeg_bad_tr = eeg_mat['OUT']['bad_TR']
            fmri_index_list = fmri_data.index.to_list()
            tr_label = np.array([0 if x+1 not in eeg_bad_tr[0][0][0] else 1 for x in fmri_index_list]).reshape(len(fmri_index_list), 1)
            tr_windows = np.lib.stride_tricks.sliding_window_view(tr_label, window_shape=(window_size, 1))[::step_size, :, :]
            input_tr = tr_windows.reshape(tr_windows.shape[0], window_size)
            binary_sums_tr = input_tr.sum(axis=1)
            bad_tr = (binary_sums_tr > 0).astype(int)

            temp_fmri_data = np.array(fmri_data)
            temp_eeg_data = np.array(df_eeg_data)
            temp_clusters = np.array(clusters)
            temp_bad_tr = np.array(bad_tr)
            fmri_data_total.append(temp_fmri_data)
            eeg_data_total.append(temp_eeg_data)
            bad_tr_total.append(temp_bad_tr)
            gt_total.append(temp_clusters)
        
        eeg_data_total = np.stack(eeg_data_total, axis=0)
        gt_total = np.stack(gt_total, axis=0)
        fmri_data_total = np.stack(fmri_data_total, axis=0)
        bad_tr_total = np.stack(bad_tr_total, axis=0)

        eeg_data_np = np.array(eeg_data_total)  
        eeg_data = np.lib.stride_tricks.sliding_window_view(
            eeg_data_np, 
            window_shape=(window_size * 525,), 
            axis=1
        )[:, ::step_size * 525, :]
        eeg_data = eeg_data.reshape(-1, eeg_data.shape[2], eeg_data.shape[3])
        gt_total = gt_total.reshape(-1, 1)
        fmri_data_np = np.array(fmri_data_total)  
        fmri_data = np.lib.stride_tricks.sliding_window_view(
            fmri_data_np, 
            window_shape=(window_size,), 
            axis=1
        )[:, ::step_size, :]
        fmri_data = fmri_data.reshape(-1, fmri_data.shape[2], fmri_data.shape[3])
        bad_tr_total = bad_tr_total.reshape(-1, 1)
        indices = np.random.permutation(len(eeg_data))
        eeg_data = eeg_data[indices]
        gt_total = gt_total[indices]
        fmri_data = fmri_data[indices]
        bad_tr_total = bad_tr_total[indices]
        
        self.indices = indices
        self.eeg_data = eeg_data
        self.gt_total = gt_total
        self.fmri_data = fmri_data
        self.bad_tr_total = bad_tr_total
        print(f"self.indices: {self.indices.shape}")
        print(f"self.eeg_data: {self.eeg_data.shape}")
        print(f"self.gt_total: {self.gt_total.shape}")
        print(f"self.fmri_data: {self.fmri_data.shape}")
        print(f"self.bad_tr_total: {self.bad_tr_total.shape}")

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        ret_dict = {}
        ret_dict["eeg"] = np.array(self.eeg_data[idx])
        ret_dict["fmri"] = np.array(self.fmri_data[idx])
        ret_dict["eeg_index"] = np.array(self.gt_total[idx])
        ret_dict["bad_tr"] = np.array(self.bad_tr_total[idx])
        ret_dict["indices"] = np.array(self.indices[idx])
        return ret_dict