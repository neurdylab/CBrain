import nibabel as nib
from nilearn import datasets, input_data, image, masking
import os
import glob
import pandas as pd
from pathlib import Path
import numpy as np

def poly_drift(order, frame_times):
    order = int(order)
    pol = np.zeros((np.size(frame_times), order + 1))
    tmax = float(frame_times.max())
    for k in range(order + 1):
        pol[:, k] = (frame_times / tmax) ** k
    pol = orthogonalize(pol)
    pol = np.hstack((pol[:, 1:], pol[:, :1]))
    return pol

def orthogonalize(X):
    if X.size == X.shape[0]:
        return X
    from scipy.linalg import pinv
    for i in range(1, X.shape[1]):
        X[:, i] -= np.dot(np.dot(X[:, i], X[:, :i]), pinv(X[:, :i]))
    return X

t_r = 0.0
atlas = datasets.fetch_atlas_difumo(dimension=64, resolution_mm=2)
maps_img = atlas.maps
maps_labels = atlas.labels
roi_names = maps_labels['difumo_names']

maps_masker = input_data.NiftiMapsMasker(maps_img=maps_img,
                                         verbose=1,
                                         detrend=True,
                                         standardize=True,
                                         standardize_confounds=True,
                                         high_variance_confounds=False,
                                         )

proc_base_dir = ''
save_base_dir = ''

source_data_format = '*.nii'
motion_base_dir = ''
source_motion_format = '*.volreg_par'
save_data_format = '*.csv'

data_paths = glob.glob(os.path.join(proc_base_dir, source_data_format))
for data_path in data_paths:
    fmri_img = nib.load(data_path)
    t_r = fmri_img.header['pixdim'][4]
    break
print(t_r)

for data_path in data_paths:
    scanname = os.path.basename(data_path)[:26] 
    motion_name = scanname[0:12] + "{:04d}".format(int(scanname[12:16]) + 1)
    motion_paths = Path(motion_base_dir)
    motion_path = list(motion_paths.glob(f'{motion_name}{source_motion_format}'))[0]

    if not os.path.exists(save_base_dir):
        os.makedirs(save_base_dir)
    
    save_path = os.path.join(save_base_dir, scanname + save_data_format)
    motion_confound = pd.read_csv(motion_path, sep='  ', header=None, engine='python')
    
    time_seq = np.arange(1, len(motion_confound) + 1)
    poly_confound = poly_drift(4, time_seq)
    poly_df = pd.DataFrame(poly_confound[:, :4])
    confounds = pd.concat([motion_confound, poly_df], axis=1)

    signals = maps_masker.fit_transform(data_path, confounds=confounds)
    mean_img = image.mean_img(data_path)
    mask = masking.compute_epi_mask(mean_img)
    mask_global_signal = input_data.NiftiLabelsMasker(mask, 'global_signal',
                                                      detrend=False,
                                                      standardize=True,
                                                      standardize_confounds=True,
                                                      t_r=t_r)
    ts_global_signal_clean = mask_global_signal.fit_transform(data_path, confounds=confounds)
    ts_global_signal = mask_global_signal.fit_transform(data_path)
    df_fmri = pd.DataFrame(signals, columns=roi_names)
    df_fmri['global signal clean'] = ts_global_signal_clean
    df_fmri['global signal raw'] = ts_global_signal
    
    df_fmri.to_csv(save_path)
    print(save_path)
