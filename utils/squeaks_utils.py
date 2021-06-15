import cv2
import os
from utils.geometry_utils import Gaze_angle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import re
import pandas as pd
import matlab.engine as engine

UTILS_PATH=r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\utils'

def get_matlab():
	eng=engine.start_matlab()
	print('matlab enginer started.')
	return eng

def load_calls(eng, path:str):
	calls=eng.loadCalls(path)
	return calls

def get_squeaks_matlab(utils_path, squeak_path,fps):
	eng=get_matlab()
	s=eng.genpath(utils_path)
	eng.addpath(s,nargout=0)
	calls=load_calls(eng,squeak_path)
	box=np.array(eng.getfield(calls, 'Box')).astype('float')
	squeak_time=box[:,0] # unit: second

	squeak_frame=squeak_time*fps
	return squeak_frame.astype('int')

def is_in_window(position, window_constraint):
	a= position[0]>np.min(window_constraint[0])
	b=position[0]<np.max(window_constraint[0])
	length = len(a)
	together=np.array([True if a[i] and b[i] else False for i in range(length)])
	return together

def get_squeaks(log_path, length):
	squeaks=np.zeros([length])
	log=open(log_path,'r',encoding='UTF-8')
	stuff=log.readline()
	while stuff:
		if 'min' in stuff:
			num=re.search("\d+(\.\d+)?",stuff)
			num = float(num.group())
			sec=num*60
			index=int(sec*15/2)
			squeaks[index:index+5]=1

		stuff=log.readline()

	log.close()

	return squeaks

def get_squeaks_indices(log_path):
	indices=[]
	log=open(log_path,'r',encoding='UTF-8')
	stuff=log.readline()
	while stuff:
		if 'min' in stuff:
			num=re.search("\d+(\.\d+)?",stuff)
			num = float(num.group())
			sec=num*60
			index=int(sec*15/2)
			for i in range(5):
				indices.append(index+i)

		stuff=log.readline()

	log.close()

	return indices


def get_thumbnail_and_fps(video_path):
	cap=cv2.VideoCapture(video_path)
	fps=cap.get(cv2.CAP_PROP_FPS)
	while True:
		ret, frame = cap.read()
		if ret:
			return frame,fps

# temporary solution for plotting squeak
def draw_squeak(root_path,squeak_path):
	root_name=os.path.split(root_path)[1]
	items=os.listdir(root_path)
	pose_file=[i for i in items if 'DLC' in i and '.csv' in i and '17391304' in i]
	video_path = os.path.join(root_path, 'camera_17391304.MOV')
	pose_path = os.path.join(root_path, pose_file[0])
	#gaze_path = os.path.join(root_path, 'gaze', 'gaze_angle_0.mat')

	pose = pd.read_csv(pose_path, header=[1, 2])
	image,fps = get_thumbnail_and_fps(video_path)
	indices=get_squeaks_matlab(UTILS_PATH,squeak_path,fps)

	indices=[i for i in indices if i<len(pose)]

	pose_snapshot = pose.loc[indices]
	length = pose_snapshot.shape[0]

	pos_x = (pose_snapshot['leftear']['x'] + pose_snapshot['rightear']['x']) / 2
	pos_y = (pose_snapshot['leftear']['y'] + pose_snapshot['rightear']['y']) / 2

	pos_x = np.array(pos_x)
	pos_y = np.array(pos_y)

	snout_x = np.array(pose_snapshot['snout']['x'])
	snout_y = np.array(pose_snapshot['snout']['y'])

	pseudo_colorbar = np.linspace(0, 255, length)
	for i in range(length):
		start = (int(pos_x[i]), int(pos_y[i]))
		end = (int(snout_x[i]), int(snout_y[i]))
		image = cv2.arrowedLine(image, start, end, color=(255, 125, 250), thickness=2)
		image = cv2.circle(image, start, radius=6, color=(50, int(pseudo_colorbar[i]), 125), thickness=-1)
	image = cv2.putText(image, root_name, org=(100, 80), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5,
						color=(255, 255, 255))
	cv2.imwrite(os.path.join(root_path, 'gaze', 'where_are_squeaks.png'), image)

if __name__=='__main__':
	#from utils.squeaks_utils import draw_squeak
	squeak_path = r'C:\Users\SchwartzLab\MatlabProjects\Detections\B&K_audio Jun-05-2021 10_16 AM.mat'
	root_path = r'D:\Desktop\2021-06-04_p6_pups_and_mama'
	draw_squeak(root_path, squeak_path)

if __name__=='__main_':
	import os
	from shutil import copyfile

	base_path = r'D:\Desktop'
	save_path = r'C:\Users\SchwartzLab\Desktop\where_are_squeaks'
	items = os.listdir(base_path)
	for item in items:

		if 'T' in item:
			squeaks_path = os.path.join(base_path, item, 'gaze', 'where_are_squeaks.png')
			copy_name = item + '.png'
			copy_path = os.path.join(save_path, copy_name)
			copyfile(squeaks_path, copy_path)
















