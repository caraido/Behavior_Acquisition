import scipy.io as sio
import numpy as np
import os
import pandas as pd
from scipy.io import wavfile

def reorganize_mat_file(rootpath,load_squeak_time=None):
	# Note that load_squeak_time is a function
	gazepath=os.path.join(rootpath, 'gaze')
	mono_view_path = os.path.join(gazepath, 'gaze_angle_32.mat')
	bino_view_path=os.path.join(gazepath, 'gaze_angle_0.mat')

	items=os.listdir(rootpath)
	dlc_file=[]
	reconstruct_file=[]
	reproject_file=[]
	squeak_file=[]


	for item in items:
		if 'DLC' in item and 'csv' in item:
			dlc_file.append(os.path.join(rootpath,item))
		if 'kalman' in item:
			reconstruct_file.append(os.path.join(rootpath,item))
		if 'reproject' in item and '.csv' in item:
			reproject_file.append(os.path.join(rootpath,item))
		if 'squeaks.mat' in item:
			squeak_file.append(os.path.join(rootpath,item))

	reproject_file = list(sorted(reproject_file))

	# if dlc data are reorganized
	if len(dlc_file)==0 and os.path.exists(os.path.join(rootpath, 'DLC')):
		dlc_path = os.path.join(rootpath, 'DLC')
		dlc_items = os.listdir(dlc_path)
		for dlc_item in dlc_items:
			if '.csv' in dlc_item:
				dlc_file.append(os.path.join(dlc_path, dlc_item))
	dlc_file = list(sorted(dlc_file))

	# load dlc data
	dlc_dict = {}
	for i,item in enumerate(dlc_file):
		data=pd.read_csv(item,header=[1,2,3])
		if 'individuals' not in data.columns[0]:
			data = pd.concat([pd.DataFrame({p:p[2] for p in data.columns}, index=[0]), data]).reset_index(drop = True)
			data.columns = data.columns.droplevel(2)

		data = data.to_dict()
		new_data={}
		for key, value in data.items(): # here value is another dict
			new_key=f'{key[0]}_{key[1]}'
			new_data[new_key]=np.array(list(value.values())).astype(np.float32)
		dlc_dict[f"camera_{i}"]=new_data

	# if reproject data are reorganized
	if len(reproject_file)==0 and os.path.exists(os.path.join(rootpath, 'reproject')):
		reproject_path = os.path.join(rootpath,'reproject')
		reproject_items=os.listdir(reproject_path)
		for reproject_item in reproject_items:
			if '.csv' in reproject_item:
				reproject_file.append(os.path.join(reproject_path,reproject_item))
	reproject_file=list(sorted(reproject_file))

	# load reproejction data
	reproject_dict={}
	for i,item in enumerate(reproject_file):
		data = pd.read_csv(item, header=[0,1],index_col=0).to_dict()
		for key, value in data.items(): # here value is another dict
			data[key]=np.array(list(value.values())).astype(np.float32)
		reproject_dict[f"camera_{i}"]=data

	if len(reconstruct_file):
		# TODO: this seems to be wrong
		reconstruct_data=pd.read_csv(reconstruct_file[0])
		reconstruct_data=reconstruct_data.to_dict()
		for key, value in reconstruct_data.items():  # here value is another dict
			reconstruct_data[key] = np.array(list(value.values())).astype(np.float32)
		reconstruct_data.pop('Unnamed: 0')
	else:
		reconstruct_data={}

	# if squeak data are organized
	if len(squeak_file)==0 and os.path.exists(os.path.join(rootpath, 'audio')):
		squeak_path = os.path.join(rootpath, 'audio')
		squeak_items = os.listdir(squeak_path)
		for squeak_item in squeak_items:
			if 'squeaks.mat' in squeak_item:
				squeak_file.append(os.path.join(squeak_path, squeak_item))

	squeaks_dict={}
	for item in squeak_file:
		squeaks=load_squeak_time(item)
		name=os.path.split(item)
		name=name[1][:-4]
		squeaks_dict[name]=squeaks

	mono_view_data = sio.loadmat(mono_view_path)
	bino_view_data=sio.loadmat(bino_view_path)

	mono_view_data=get_gaze_data(mono_view_data)
	bino_view_data=get_gaze_data(bino_view_data)

	full_data={
		'mono_gaze':mono_view_data,
		'bino_gaze':bino_view_data,
		'DLC_tracking':dlc_dict,
		'reprojection':reproject_dict,
		'reconstruction':reconstruct_data,
		'squeaks_time':squeaks_dict
	}

	savepath=os.path.join(rootpath,'full_data.mat')
	sio.savemat(savepath,full_data, long_field_names=True)
	print('saved full data')

'''
def reorganize_mat_file2(rootpath):
	gazepath=os.path.join(rootpath, 'gaze')
	mono_view_path = os.path.join(gazepath, 'gaze_angle_32.mat')
	bino_view_path=os.path.join(gazepath, 'gaze_angle_0.mat')


	items=os.listdir(rootpath)
	dlc_file=[]
	reconstruct_file=[]
	reproject_file=[]
	audio_file=[]

	dlc_path = os.path.join(rootpath, 'DLC')
	dlc_items = os.listdir(dlc_path)
	for dlc_item in dlc_items:
		if '.csv' in dlc_item:
			dlc_file.append(os.path.join(dlc_path, dlc_item))

	for item in items:
		#if 'DLC' in item and 'csv' in item:
		#	dlc_file.append(os.path.join(rootpath,item))
		if 'kalman' in item:
			reconstruct_file.append(os.path.join(rootpath,item))
		if 'reproject' in item and '.csv' in item:
			reproject_file.append(os.path.join(rootpath,item))

	dlc_file=list(sorted(dlc_file))
	reproject_file = list(sorted(reproject_file))

	dlc_dict={}
	for i,item in enumerate(dlc_file):
		data=pd.read_csv(item,header=[1,2,3]).center_mouse
		data = data.to_dict()
		new_data={}
		for key, value in data.items(): # here value is another dict
			new_key=f'{key[0]}_{key[1]}'
			new_data[new_key]=np.array(list(value.values())).astype(np.float32)
		dlc_dict[f"camera_{i}"]=new_data

	reproject_dict={}
	for i,item in enumerate(reproject_file):
		data=pd.read_csv(item)
		data=data.to_dict()
		data.pop('Unnamed: 0')
		for key, value in data.items(): # here value is another dict
			data[key]=np.array(list(value.values())).astype(np.float32)
		reproject_dict[f"camera_{i}"]=data

	try:
		reconstruct_data=pd.read_csv(reconstruct_file[0])
		reconstruct_data=reconstruct_data.to_dict()
		for key, value in reconstruct_data.items():  # here value is another dict
			reconstruct_data[key] = np.array(list(value.values())).astype(np.float32)
		reconstruct_data.pop('Unnamed: 0')
	except:
		reconstruct_data={}

	mono_view_data = sio.loadmat(mono_view_path)
	bino_view_data=sio.loadmat(bino_view_path)

	mono_view_data=get_gaze_data(mono_view_data)
	bino_view_data=get_gaze_data(bino_view_data)

	full_data={
		'mono_gaze':mono_view_data,
		'bino_gaze':bino_view_data,
		'DLC_tracking':dlc_dict,
		'reprojection':reproject_dict,
		'reconstruction':reconstruct_data,
	}

	savepath=os.path.join(rootpath,'full_data.mat')
	sio.savemat(savepath,full_data)
	print('saved full data')
'''

def get_gaze_data(matfile:dict):
	triangle = {'body':matfile['body_triangle_area'][0],
					'head': matfile['head_triangle_area'][0]
				}
	gaze = {'inner_wall': {'left': matfile['inner_left'][0],
						   'right': matfile['inner_right'][0]},
			'outer_wall':{'left':matfile['outer_left'][0],
						  'right':matfile['outer_right'][0]},
			}
	accumulative_gaze = {'outer_wall':{'left':matfile['left_accum'],
									   'right':matfile['right_accum']}
						 }
	accumulative_body_position = {'outer_wall':matfile['body_accum']
	}
	new_data={'triangle_area':triangle,
			  'gaze':gaze,
			  'accumulative_gaze':accumulative_gaze,
			  'accumulative_body_position':accumulative_body_position,
			  'speed': matfile['speed'][0],
			  'body_position_arc':matfile['body_position'][0],
			  'stats':matfile['stats'],
			  'window_visibility':matfile['win_visibility'].transpose(),
			  'nose_window_distance':matfile['nose_window_distance'],
			  'window_crossings':matfile['window_crossings']
	}
	return new_data

def add_new_gaze_to_full_mat(from_gaze_path,full_data_path):
	# gaze path should be a folder
	# full data paths should be a .mat file
	full_data=sio.loadmat(full_data_path)
	gaze0=sio.loadmat(os.path.join(from_gaze_path,'gaze_angle_0.mat'))
	gaze32=sio.loadmat(os.path.join(from_gaze_path,'gaze_angle_32.mat'))
	bino_gaze=get_gaze_data(gaze0)#gaze0
	mono_gaze=get_gaze_data(gaze32)#gaze32
	full_data['bino_gaze']=bino_gaze
	full_data['mono_gaze']=mono_gaze
	sio.savemat(full_data_path,full_data)
	print('added new gaze data to full data mat')
