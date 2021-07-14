import cv2
import os
from utils.geometry_utils import Gaze_angle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import re
import pandas as pd
import matlab.engine

UTILS_PATH=r'C:\\Users\\SchwartzLab\\PycharmProjects\\bahavior_rig\\utils'
DEEPSQUEAK_PATH=r"C:\\Users\\SchwartzLab\\MatlabProjects"
default_network='All Short Calls_Network_V1.mat'
SETTINGS=[0,8,0.002,80,35,0,1]

class Squeaks:

	def __init__(self):
		self.engine=matlab.engine.start_matlab()
		self.network_path=default_network
		print('matlab engine start')
		self.root_path=None
		self.settings=matlab.double(SETTINGS)

	def set_root_path(self,path):
		self.root_path=path

	def set_settings(self,new_settings):
		if len(new_settings)==6:
			self.settings=matlab.double(new_settings)
		else:
			raise Exception('wrong number of settings')

	def __call__(self,fname):
		# save squeaks and return squeak time in second
		# fname being the wav file of the audio
		if root_path is not None:
			print(f'currently analyzing squeaks under: {self.root_path}')

			calls=self.engine.GetSqueaks(DEEPSQUEAK_PATH,
										  self.root_path,
										  fname,
										  self.network_path,
										  self.settings
										  )

			if calls:
				box = np.array(self.engine.getfield(calls, 'Box')).astype('float')
				squeak_time=box[:,0]
			else:
				squeak_time=None
			print('saved calls')
			self.engine.clear
			return squeak_time

	def load_squeak_time_matlab(self,squeak_path):
		calls=self.engine.loadCallfile(squeak_path)
		try:
			box=np.array(self.engine.getfield(calls, 'Box')).astype('float')
			squeak_time=box[:,0]
		except:
			squeak_time=None
			print('no squeaks in the file')
		return squeak_time

	def stop_matlab(self):
		self.engine.quit()
		print('background matlab stopped')

	def restart_matlab(self):
		self.engine.start_matlab()

	def __del__(self):
		self.stop_matlab()

	def draw_BK_squeaks(self,root_path,squeak_time=None):
		root_name = os.path.split(root_path)[1]
		items = os.listdir(root_path)
		pose_file = [i for i in items if 'DLC' in i and '.csv' in i and '17391304' in i]
		if len(pose_file)==0 and os.path.exists(os.path.join(root_path,'DLC')):
			dlcpath=os.path.join(root_path,'DLC')
			subitems=os.listdir(dlcpath)
			pose_file = [i for i in subitems if 'DLC' in i and '.csv' in i and '17391304' in i]
			pose_path = os.path.join(dlcpath,pose_file[0])
		else:
			pose_path = os.path.join(root_path, pose_file[0])
		try:
			video_path = os.path.join(root_path, 'camera_17391304.MOV')
		except:
			video_path=os.path.join(root_path,'raw','camera_17391304.MOV')

		pose = pd.read_csv(pose_path, header=[1, 2])
		image,fps = get_thumbnail_and_fps(video_path)

		if squeak_time is None:
			try:
				squeak_time=self.load_squeak_time_matlab(os.path.join(root_path,'B&K_audio.mat'))
			except:
				raise Exception('squeak time file not found! Please analyze the squeaks first')

		if squeak_time is not None:
			all_indices=squeak_time*fps

			indices = [i for i in all_indices if i < len(pose)]# let's see how python handle empty squeak file

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
		else:
			image = cv2.putText(image,'No squeak detected!',org=(500,80), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2,
								color=(255, 255, 255))
		cv2.imwrite(os.path.join(root_path, 'gaze', 'where_are_squeaks_B&K.png'), image)

	def draw_Dodo_squeaks(self, root_path, squeak_time=None):
		root_name = os.path.split(root_path)[1]
		items = os.listdir(root_path)
		pose_file = [i for i in items if 'DLC' in i and '.csv' in i and '17391304' in i]
		if len(pose_file) == 0 and os.path.exists(os.path.join(root_path, 'DLC')):
			dlcpath = os.path.join(root_path, 'DLC')
			subitems = os.listdir(dlcpath)
			pose_file = [i for i in subitems if 'DLC' in i and '.csv' in i and '17391304' in i]
			pose_path = os.path.join(dlcpath, pose_file[0])
		else:
			pose_path = os.path.join(root_path, pose_file[0])

		video_path = os.path.join(root_path, 'camera_17391304.MOV')
		if not os.path.exists(video_path):
			video_path = os.path.join(root_path, 'raw', 'camera_17391304.MOV')

		pose = pd.read_csv(pose_path, header=[1, 2])
		image, fps = get_thumbnail_and_fps(video_path)

		if squeak_time is None:
			try:
				squeak_time = self.load_squeak_time_matlab(os.path.join(root_path, 'Dodo_audio.mat'))
			except:
				raise Exception('squeak time file not found! Please analyze the squeaks first')

		if squeak_time is not None:
			all_indices = squeak_time * fps
			all_indices=all_indices.astype('int')


			indices = [i for i in all_indices if i < len(pose)]  # let's see how python handle empty squeak file

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
		else:
			image = cv2.putText(image, 'No squeak detected!', org=(500, 80), fontFace=cv2.FONT_HERSHEY_DUPLEX,
								fontScale=2,
								color=(255, 255, 255))
		if os.path.exists(os.path.join(root_path, 'gaze')):
			cv2.imwrite(os.path.join(root_path, 'gaze', 'where_are_squeaks_Dodo.png'), image)
		else:
			cv2.imwrite(os.path.join(root_path, 'where_are_squeaks_Dodo.png'), image)

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


if __name__=='__main__':
	root_path=r'D:\\Desktop\\archives\\2021-06-03_alec_testing_rats'
	squeaks=Squeaks()
	squeaks.set_root_path(root_path)
	fname='Dodo_audio.wav'
	# detect the call
	call_time=squeaks(fname)
	# load the call
	#squeak_path=os.path.join(root_path,'Dodo_audio_squeaks.mat')
	#call_time=squeaks.load_squeak_time_matlab(squeak_path)
	# draw squeaks and save
	squeaks.draw_Dodo_squeaks(root_path,call_time)


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
















