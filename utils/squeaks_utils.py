import cv2
import numpy as np
import pandas as pd
import matlab.engine
from global_settings import UTILS_PATH, DEEPSQUEAK_PATH,default_network,SETTINGS
import os


class Squeaks:

	def __init__(self):
		self.engine=matlab.engine.start_matlab()
		s = self.engine.genpath(UTILS_PATH)
		self.engine.addpath(s, nargout=0)
		self.network_path=default_network
		print('matlab engine is started')
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
		if self.root_path is not None:
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
				squeak_time=False
			print('saved calls')
			self.engine.clear
			return squeak_time

	def load_squeak_time_matlab(self,squeak_path):
		calls=self.engine.loadCallfile(squeak_path)
		try:
			box=np.array(self.engine.getfield(calls, 'Box')).astype('float')
			squeak_time=box[:,0]
		except:
			squeak_time=False
			print('no squeaks in the file')
		return squeak_time

	def stop_matlab(self):
		self.engine.quit()
		print('background matlab stopped')

	def restart_matlab(self):
		self.engine.start_matlab()
		print('background matlab restarted')

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

		video_path = os.path.join(root_path, 'camera_17391304.MOV')
		if not os.path.exists(video_path):
			video_path = os.path.join(root_path, 'raw', 'camera_17391304.MOV')

		pose = pd.read_csv(pose_path, header=[1, 2])
		image,fps = get_thumbnail_and_fps(video_path)

		if squeak_time is None:
			try:
				squeak_time=self.load_squeak_time_matlab(os.path.join(root_path,'B&K_audio.mat'))
			except:
				raise Exception('squeak time file not found! Please analyze the squeaks first')

		if not isinstance(squeak_time,int):
			all_indices=squeak_time*fps

			indices = [int(i) for i in all_indices if i < len(pose)]# let's see how python handle empty squeak file

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

		cv2.imwrite(os.path.join(root_path, 'where_are_squeaks_BK.png'), image)

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

		if not isinstance(squeak_time,int):
			all_indices = squeak_time * fps

			indices = [int(i) for i in all_indices if i < len(pose)]  # let's see how python handle empty squeak file

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
		cv2.imwrite(os.path.join(root_path, 'where_are_squeaks_Dodo.png'), image)

	def draw_spectrogram(self,working_path,saving_path):
		status=self.engine.GetSpectrogram(working_path,saving_path)
		if not status:
			print("not squeaks detected so no spectrogram.")

def get_thumbnail_and_fps(video_path):
	cap=cv2.VideoCapture(video_path)
	fps=cap.get(cv2.CAP_PROP_FPS)
	while True:
		ret, frame = cap.read()
		if ret:
			cap.release()
			del cap
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
















