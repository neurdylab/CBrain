clear
clc

proc_base_dir = '';
save_base_dir = '';

proc_dirs = {};

source_file_format = '*.mat';

for pp = 1:length(proc_dirs)
    source_files = dir([proc_base_dir, proc_dirs{pp}, source_file_format]);
    for i = 1:length(source_files)
        file_name = source_files(i).name;
        file_name = erase(file_name, '.mat');
        load([source_files(i).folder, '/', source_files(i).name]);
        EEG_info(i).scan = file_name;
        EEG_info(i).frames_bufferOv = frames_bufferOv;
        EEG_info(i).bad_channels = bad_channels;
        save_dir = [save_base_dir, proc_dirs{pp}];
        if ~exist(save_dir, 'dir')
            mkdir(save_dir)
        end
        pop_saveset(EEG,'filename',[file_name,'.set'],'filepath',save_dir);
        clear EEG EEG_info frames_bufferOv bad_channels
    end
end


