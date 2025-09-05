clear
clc

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

proc_base_dir = '';
save_base_dir = '';

seteeg_list = dir(fullfile(proc_base_dir, '*.set'));
for i = 1:length(seteeg_list)
    file_name = seteeg_list(i).name;
    EEG = pop_loadset('filename', file_name, 'filepath', seteeg_list(i).folder);
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
    EEG = pop_select( EEG, 'rmchannel',{'EOG','ECG'});
    new_file_name = [erase(file_name, '.mat'), '_30.set'];
    save_path = [save_base_dir, new_file_name];
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'setname','eeg_trim_30','savenew', save_path,'comments',strvcat('Original file: ecr_run1_575frames_cbc.dat','Removed channels:','EOG','ECG',' ','Result channel number: 30'),'gui','off');
    clear EEG EEG_info
end
