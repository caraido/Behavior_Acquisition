function Calls=GetSqueaks(working_path,root_path,fname,networkname,Settings)

    % get to the woring path
    cd(working_path);% working_path has to be the full path

    % Get inputputfile
    load_file_name =fullfile(root_path,fname); %item_name being a .wav file
    sname=append(fname(1:end-4),'_squeaks.mat');
    save_file_name=fullfile(root_path,sname);

    currentFile=1;
    totalFiles=1;
    number_of_repeats=2;
    network_path=fullfile(working_path,'Networks',networkname);
    networkfile=load(network_path);

    % detect squeaks
    Calls=SqueakDetect(load_file_name,networkfile, fname, Settings, currentFile,totalFiles,networkname,number_of_repeats);
    % save to .mat file
    save(save_file_name,'Calls');

    if height(Calls)==0
        Calls=0;
    end


