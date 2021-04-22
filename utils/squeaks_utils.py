import cv2
import os
from utils.geometry_utils import Gaze_angle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import re
import pandas as pd

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

def get_thumbnail(video_path):
	cap=cv2.VideoCapture(video_path)

	while True:
		ret, frame = cap.read()
		if ret:
			return frame



if __name__ =='__main_':
	path =  r'D:\Desktop'
	roots=os.listdir(path)
	for root in roots:
		if "T" in root:
			root_path = os.path.join(path,root)
			gaze_path = os.path.join(root_path,'gaze')
			items=os.listdir(gaze_path)
			gaze_files=[os.path.join(gaze_path,item) for item in items if '.mat' in item]
			config_path = os.path.join(root_path,'config')
			gaze_model =Gaze_angle(config_path)


			plt.figure(figsize=(10, 15))
			plt.title(root)

			for i,gaze in enumerate(gaze_files):
				gaze0=sio.loadmat(gaze)
				# get gaze
				winA=gaze_model.windows[0]
				winB=gaze_model.windows[1]
				winC = gaze_model.windows[2]

				body_parts=['body_position','outer_left','outer_right']
				for j,body_part in enumerate(body_parts):

					body_part_in_A = is_in_window(gaze0[body_part],winA)
					body_part_in_B = is_in_window(gaze0[body_part], winB)
					body_part_in_C = is_in_window(gaze0[body_part], winC)

					# get squeaks
					log_path=os.path.join(root_path,'log.txt')
					length = len(body_part_in_C)


					# body position
					if body_part=='body_position' and 'gaze_angle_0' in gaze:
						continue
					if body_part=='outer_left' and 'gaze_angle_0' in gaze:
						continue

					plt.subplot(5,1,i*3+j-1)
					if i*3+j-1==1:
						plt.title(root)
					plt.eventplot(np.where(body_part_in_A),colors='red')
					plt.eventplot(np.where(body_part_in_B),colors='green')
					plt.eventplot(np.where(body_part_in_C),colors='blue')
					plt.legend(['A','B','C'])
					plt.xlim([0,length])
					plt.xticks([])
					name = os.path.split(gaze)[1][:-4]
					if body_part=='outer_right' and 'gaze_angle_0' in gaze:
						body_part='outer_head_direction'
					plt.ylabel(name+'_'+body_part)

			plt.subplot(5,1,5)
			squeaks = get_squeaks(log_path=log_path, length=length)
			plt.eventplot(np.where(squeaks),colors='gray')
			plt.xlim([0,length])
			plt.xlabel('frame')
			plt.ylabel('squeaks')

			plt.tight_layout()

			plt.savefig(os.path.join(gaze_path,'gaze_and_squeaks.png'))

if __name__=='__main_':
	import tqdm
	img_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\test.png'

	all_path=r'D:\Desktop'
	items=os.listdir(all_path)
	for item in tqdm.tqdm(items):
		if "T" in item:
			root_path=os.path.join(all_path,item)
			video_path = os.path.join(root_path,'camera_17391304.MOV')
			pose_path = os.path.join(root_path,'camera_17391304DLC_resnet50_Alec_second_tryDec7shuffle1_120000.csv')
			gaze_path = os.path.join(root_path,'gaze','gaze_angle_0.mat')
			log_path=os.path.join(root_path,'log.txt')

			pose=pd.read_csv(pose_path,header=[1,2])
			gaze=sio.loadmat(gaze_path)
			image = get_thumbnail(video_path)
			#image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
			indices=get_squeaks_indices(log_path)

			pose_snapshot=pose.loc[indices]
			length = pose_snapshot.shape[0]

			pos_x=(pose_snapshot['leftear']['x']+pose_snapshot['rightear']['x'])/2
			pos_y = (pose_snapshot['leftear']['y'] + pose_snapshot['rightear']['y']) / 2

			pos_x=np.array(pos_x)
			pos_y = np.array(pos_y)

			snout_x=np.array(pose_snapshot['snout']['x'])
			snout_y = np.array(pose_snapshot['snout']['y'])

			pseudo_colorbar=np.linspace(0,255,length)

			for i in range(length):
				start=(int(pos_x[i]),int(pos_y[i]))
				end = (int(snout_x[i]), int(snout_y[i]))
				image=cv2.arrowedLine(image,start,end,color=(255,125,250),thickness=2)
				image=cv2.circle(image,start,radius=5,color=(50,int(pseudo_colorbar[i]),125),thickness=-1)
			image=cv2.putText(image,item,org=(300,80),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=2.3,color=(255,255,255))
			cv2.imwrite(os.path.join(root_path,'gaze','where_are_squeaks.png'),image)

if __name__=='__main__':
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
















