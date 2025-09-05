clear
clc

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

proc_base_dir = '';
save_base_dir = '';

proc_dirs = {};

source_file_format = '*.set';

for pp = 1:length(proc_dirs)
    source_files = dir([proc_base_dir, proc_dirs{pp}, source_file_format]);
    for i = 1:length(source_files)
        file_name = source_files(i).name;
        EEG = pop_loadset('filename', file_name, 'filepath', source_files(i).folder);
        [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
        EEG = pop_select( EEG, 'rmchannel',{'EOG1','EOG2','EMG1','EMG2','EMG3','ECG', 'CWL1', 'CWL2', 'CWL3', 'CWL4'});
        save_dir = [save_base_dir, proc_dirs{pp}];
        if ~exist(save_dir, 'dir')
            mkdir(save_dir)
        end
        new_file_name = [erase(file_name, '.mat'), '_26.set'];
        save_path = [save_dir, new_file_name];
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','eeg_trim_26','savenew', save_path,'comments',strvcat('Original file: ecr_run1_575frames_cbc.dat','Removed channels:','EOG1 27','EOG2 28','EMG1 29','EMG2 30','EMG3 31','ECG 32',' ','Result channel number: 26'),'gui','off');
        clear EEG 
    end
end