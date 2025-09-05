clear
clc

proc_base_dir = '';
save_base_dir = '';

raweeg_list = dir(fullfile(proc_base_dir, '*.mat'));
for i = 1:length(raweeg_list)
    file_name = raweeg_list(i).name;
    file_name = erase(file_name, '.mat');
    raweeg_path = fullfile(proc_base_dir, raweeg_list(i).name);
    load(raweeg_path);
    EEG_info(i).scan = file_name;
    save_dir = save_base_dir;
    pop_saveset(EEG,'filename',[file_name,'.set'],'filepath',save_dir);
    clear EEG EEG_info 
end


