import os
import re
import time
import shutil
import toml
import json
from utils.datajoint_utils import DjConn
from global_settings import saving_path_prefix,default_saving_path,default_folder_name,GLOBAL_CONFIG_PATH,GLOBAL_CONFIG_ARCHIVE_PATH,global_log_path,namespace_path

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

def get_extrinsic_path(camera:list,config_path=GLOBAL_CONFIG_PATH):
	path=list(map(lambda x: os.path.join(config_path,'config_extrinsic_%s_temp.MOV'%x),camera))
	path.append(None)
	path.append(None)
	return path


def separate_name(name):
	if name[0]=='&':
		result = True
		name_list=re.split(r'[&]',name[1:])
		if len(name_list[0])==0: # empty animal ID
			name_list[0]='unnamedAnimal'
			result = False
		if len(name_list[1]) == 0: # empty animaltype
			name_list[1]='none'
		if len(name_list[2]) == 0: # empty sessiontype
			name_list[2] ='none'
		if len(name_list[3]) ==0: # emtpy window A
			name_list[3]='empty'
		if name_list[3] == 'empty':
			name_list[4]='' # window A DJID ==0
		if len(name_list[5]) == 0 :  # emtpy window B
			name_list[5] = 'empty'
		if name_list[5] == 'empty':
			name_list[6]='' # window B DJID==0
		if len(name_list[7]) ==0 : # emtpy window C
			name_list[7]='empty'
		if name_list[7]=='empty':
			name_list[8]=''
		return result,name_list
	else:
		raise NameError('Wrong naming system!')


def update_namespace(namelist):
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


def add_new_type_to_datajoint(namelist):
	conn = DjConn()
	connected = conn.connect_to_datajoint()
	print(connected)
	# conn.get_main_schema()
	status,info=conn.add_new_type_to_dj(namelist)
	return status,info,conn


def reformat_filepath(path,name,camera:list):

	date= time.strftime("_%Y-%m-%d_",time.localtime())
	if path == '':
		real_path = os.path.join(saving_path_prefix,default_saving_path)
		print("No file path specified. Will use default path")
	else:
		real_path = os.path.join(saving_path_prefix,path)

	if not os.path.exists(real_path):
		os.makedirs(real_path)
		print("file path %s doesn't exist, creating one..." % real_path)

	result,namelist=separate_name(name)
	subfolder=os.path.join(real_path,namelist[0]) # namelist[0] is animal ID

	if result:
		result,info_type,conn=add_new_type_to_datajoint(namelist)

	if namelist[0] == 'unnamedAnimal':
		sessID = 0 # not updating db...
		info = ''
	elif result:
		sessID,info=conn.update_session(namelist)
		if not sessID:
			return False,None,info_type+'\n'+info #there was an error, so return and don't record anything
	else:
		return False,None,info_type
		# sessID = 0 #in case we are not trying to update the database... ?
			
	if not os.path.exists(subfolder): #make the folder for this animal
		os.mkdir(subfolder)
		print("file path %s doesn't exist, creating one..." % real_path)

	#next, set up the folder for this session
	condition2=namelist[2]=='habituation'
	if condition2:
		full_path = os.path.join(subfolder,str(sessID)+date+namelist[1]+'_habituation')
	else:
		full_path=os.path.join(subfolder,str(sessID)+date+namelist[1]+'_'+namelist[2]+r"_(A)"+namelist[3]+r"_(B)"+namelist[5]+r"_(C)"+namelist[7])

	# for repeated sessions, the following method overwrite the original
	# TODO: is this ever invoked?
	if os.path.exists(full_path):
		shutil.rmtree(full_path)
		os.mkdir(full_path)
	else:
		os.mkdir(full_path)

	filepaths = []
	for serial_number in camera:
		camera_filepath = os.path.join(full_path,'camera_%s.MOV'%serial_number)
		filepaths.append(camera_filepath)

	mic_filepath=os.path.join(full_path,'Dodo_audio.tdms')
	filepaths.append(mic_filepath)

	audio_filepath = os.path.join(full_path,'B&K_audio.tdms')
	filepaths.append(audio_filepath)

	# TODO: need to implement the handling in GUI to popup a new alert window indicating the status
	return True,filepaths,info_type+'\n'+info

def copy_config(filepath,version=None):
	# version should be int
	# TODO：version control not implemented. Now it always saves the latest version
	local_config_path = os.path.join(filepath, 'config')
	archive_items = os.listdir(GLOBAL_CONFIG_ARCHIVE_PATH)
	versions = [(int(re.findall(r"_v(\d+)_", i)[0]), i) for i in archive_items]
	versions = sorted(versions)

	if version is None:
		version_fullname=versions[-1][1]
		this_version=versions[-1][0]
	else:
		version_numbers=[a[0] for a in versions]
		if version not in version_numbers:
			raise Exception("the version number of configuration file is not found")
		else:
			this_version=version
			version_fullname=versions[version][1]

	config_path = os.path.join(GLOBAL_CONFIG_ARCHIVE_PATH,version_fullname)
	if not os.path.exists(local_config_path):
		shutil.copytree(config_path, local_config_path)

	return this_version


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

def main():
	# filename = '&1190&male_unknown_social_status&habituation&&&&&&&Devon&'
	# filename = '&1207&alec_testing&alec_testing&&&&&&&Zach&'
	filename = '&1207&alec_testing&alec_testing&juvenile&1190&&&&&Zach&'
	
	camera_list = [17391304, 17391290, 21259803, 19412282]

	reformat_filepath('', filename, camera_list)

if __name__ == '__main__':
	main()








