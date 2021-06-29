import os
import re
import time
import shutil
import toml
import json

saving_path_prefix = 'D:\\'
default_saving_path= 'Desktop'
default_folder_name = 'Testing'
global_config_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\config'
global_log_path=r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\log'
namespace_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\behavior_gui\assets\namespace\namespace.json'

@property
def get_saving_path_prefix():
	return saving_path_prefix

@property
def get_default_path():
	return default_saving_path

@property
def get_default_name():
	return default_folder_name

def change_default_path(input_path):
	global default_saving_path
	if os.path.exists(os.path.join(saving_path_prefix, default_saving_path)):
		default_saving_path = input_path
		print('changed default saving path into: '+ input_path)
	else:
		raise NotADirectoryError("The specified path doesn't exist!")

def change_default_name(input_name):
	global default_folder_name
	default_folder_name=input_name
	print('changed default saving folder name into: ' + input_name)

def change_path_prefix(input_prefix):
	global saving_path_prefix
	if os.path.exists(os.path.normpath(saving_path_prefix)):
		saving_path_prefix = input_prefix
	else:
		raise NotADirectoryError("The specified path doesn't exist!")

def get_extrinsic_path(camera:list,config_path=global_config_path):
	path=list(map(lambda x: os.path.join(config_path,'config_extrinsic_%s_temp.MOV'%x),camera))
	path.append(None)
	path.append(None)
	return path

def seperate_name(name):
	if name[0]=='&':
		name_list=re.split(r'[&]',name[1:])
		if len(name_list[0])==0: # empty animal name
			name_list[0]='unnamedAnimal'
		if len(name_list[1]) ==0: # empty animaltype
			name_list[1]='unknownAnimalType'
		if len(name_list[2]) ==0: # emtpy window A
			name_list[2]='None'
		if len(name_list[3]) == 0:  # emtpy window B
			name_list[3] = 'None'
		if len(name_list[4]) ==0: # emtpy window C
			name_list[4]='None'
		return name_list
	else:
		raise NameError('Wrong naming system!')

def add_to_namespace(namelist):
	animalID = namelist[0]
	animalType = namelist[1]
	windows = namelist[2:]  # list
	with open(namespace_path,'r') as f:
		namespace=json.load(f)
		if animalID not in namespace['animalID']:
			namespace['animalID'].append(animalID)
			namespace['animalID']=list(sorted(namespace['animalID']))
		if animalType not in namespace['animalType']:
			namespace['animalType'].append(animalType)
			namespace['animalType'] = list(sorted(namespace['animalType']))
		namespace['windows']=namespace['windows']+windows
		namespace['windows']=list(sorted(list(set(namespace['windows']))))
	with open(namespace_path,'w') as f:
		f.seek(0)  # <--- should reset file position to the beginning.
		json.dump(namespace, f, indent=4)
		f.truncate()
		print('updated namespace.json')


def reformat_filepath(path,name,camera:list):

	date= time.strftime("%Y-%m-%d_",time.localtime())
	if path == '':
		real_path = os.path.join(saving_path_prefix,default_saving_path)
		print("No file path specified. Will use default path")
	else:
		real_path = os.path.join(saving_path_prefix,path)

	if not os.path.exists(real_path):
		os.makedirs(real_path)
		print("file path %s doesn't exist, creating one..." % real_path)

	namelist=seperate_name(name)
	subfolder=os.path.join(real_path,namelist[0]) # namelist[0] is animal ID

	add_to_namespace(namelist)

	if not os.path.exists(subfolder):
		os.mkdir(subfolder)
		print("file path %s doesn't exist, creating one..." % real_path)

	if namelist[2]=='None' and namelist[3]=='None' and namelist[4]=='None':
		full_path = os.path.join(subfolder,date+namelist[1]+'_habituation')
	else:
		full_path=os.path.join(subfolder,date+namelist[1]+r"_(A)"+namelist[2]+r"_(B)"+namelist[3]+r"_(C)"+namelist[4])

	if not os.path.exists(full_path):
		os.mkdir(full_path)
	else:
		i=2
		while True:
			if os.path.exists(full_path+'(%s)'%i):
				i+=1
			else:
				full_path=full_path+'(%s)'%i
				os.mkdir(full_path)
				break

	filepaths = []
	for serial_number in camera:
		camera_filepath = os.path.join(full_path,'camera_%s.MOV'%serial_number)
		filepaths.append(camera_filepath)

	mic_filepath=os.path.join(full_path,'Dodo_audio.tdms')
	filepaths.append(mic_filepath)

	audio_filepath = os.path.join(full_path,'B&K_audio.tdms')
	filepaths.append(audio_filepath)

	return filepaths

'''
def reformat_filepath(path,name,camera:list):
	date= time.strftime("%Y-%m-%d_",time.localtime())
	if path == '':
		real_path = os.path.join(saving_path_prefix,default_saving_path)
		print("No file path specified. Will use default path")
	else:
		real_path = os.path.join(saving_path_prefix,path)

	if not os.path.exists(real_path):
		os.makedirs(real_path)
		print("file path %s doesn't exist, creating one..." % real_path)

	if name =='':
		full_path = os.path.join(real_path, date + default_folder_name)
		print("No folder name specified. Will use default folder name")
	else:
		full_path = os.path.join(real_path, date + name)

	if not os.path.exists(full_path):
		os.mkdir(full_path)
		print("file path %s doesn't exist, creating one..." % real_path)
	else:
		i=1
		while True:
			if os.path.exists(full_path+'(%s)'%i):
				i+=1
			else:
				full_path=full_path+'(%s)'%i
				os.mkdir(full_path)
				break

	filepaths = []
	for serial_number in camera:
		camera_filepath = os.path.join(full_path,'camera_%s.MOV'%serial_number)
		filepaths.append(camera_filepath)

	mic_filepath=os.path.join(full_path,'Dodo_audio.tdms')
	filepaths.append(mic_filepath)

	audio_filepath = os.path.join(full_path,'B&K_audio.tdms')
	filepaths.append(audio_filepath)

	return filepaths
'''

def copy_config(filepath):
	local_config_path = os.path.join(filepath, 'config')
	if not os.path.exists(local_config_path):
		os.mkdir(local_config_path)
	global_configs = os.listdir(global_config_path)
	for item in global_configs:
		shutil.copy(os.path.join(global_config_path, item), local_config_path)


def load_config(filepath):
	local_config_path = os.path.join(filepath,'config')
	configs = os.listdir(local_config_path)
	data = {}
	for config in configs:
		data[config]=toml.load(os.path.join(local_config_path,config))
	return data


def save_notes(content:str, save_paths):
	# make temporary directory if not exist
	if not os.path.exists(global_log_path):
		os.makedirs(global_log_path)

	# record the time stamp
	date = time.strftime("%Y-%m-%d_[%H:%M:%S]", time.localtime())

	temp_file_name = 'temp_log.txt'
	temp_file = os.path.join(global_log_path, temp_file_name)
	file_name = 'log.txt'

	if not any(save_paths):
		if len(content)!=0:
			with open(temp_file,'w') as f:
				f.write(date+':\n'+content+'\n')
			print('saved the log')
	else:
		root_file_path = os.path.split(save_paths[0])[0]
		file = os.path.join(root_file_path, file_name)

		if len(content)!=0:
			after_recording_content=date+':\n'+content+'\n'
			if os.path.exists(temp_file):
				with open(temp_file,'r') as f:
					before_recording_note=f.read()
				entire_note=before_recording_note+after_recording_content
			else:
				entire_note=after_recording_content
		else:
			if os.path.exists(temp_file):
				with open(temp_file,'r') as f:
					before_recording_note=f.read()
				entire_note=before_recording_note
			else:
				entire_note=''

		with open(file, 'w') as f:
			f.write(entire_note)
		print('saved the log')

		try:
			os.remove(temp_file)
		except:
			pass











