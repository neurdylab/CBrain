import nibabel as nib
from nilearn import datasets, input_data, image, masking
import os
import glob
import pandas as pd

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
proc_dirs = {}

data_dir = ''
motion_dir = ''
save_dir = ''

source_data_format = '*.nii.gz'
source_motion_format = '*.volreg_par'
save_data_format = '*.csv'

for dir in proc_dirs:
    data_paths = glob.glob(os.path.join(proc_base_dir, dir, data_dir, source_data_format))
    for data_path in data_paths:
        fmri_img = nib.load(data_path)
        t_r = fmri_img.header['pixdim'][4]
        break
    break

for dir in proc_dirs:
    data_paths = glob.glob(os.path.join(proc_base_dir, dir, data_dir, source_data_format))
    for data_path in data_paths:
        scanname = os.path.basename(data_path)[:13]
        motion_path = glob.glob(os.path.join(proc_base_dir, dir, motion_dir, source_motion_format))[0]
        save_dir_path = os.path.join(save_base_dir, dir, save_dir)

        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        save_path = os.path.join(save_dir_path, scanname+save_data_format) 
        motion_confound = pd.read_csv(motion_path, sep='  ', header=None, engine='python')
        confounds = motion_confound

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