# idea: 
# should not run at same time as ag


# files/directories management
# .tdms to .mat (squeaks?)
# log of errors?
# run offline dlc model on videos 
# copy to server and hard drive
# generate plots (histogram of view angle, mouse location, maybe squeak analysis down the line)

# raw and processed subfolders

# separated into threads? each runs through a sequence of tasks
# one for video
# run dlc models on 1 video at a time?
# generate plots
# sends to server, HDD
# one for audio
# convert to .mat / .wav etc.
# down the line: run deepsqueak
# down the line: analyze squeaks and plot?
# sends to server, HDD
# one for other stuff
# compile errors/config info into a file in the directory
# calibration
# sends to server, HDD

# when all the threads are done
# delete SSD folder
import re
import threading

from utils.path_operation_utils import copy_config, global_config_path,global_config_archive_path
from utils.calibration_utils import undistort_videos,  Calib, TOP_CAM
from utils.dlc_utils import dlc_analysis,SIDE_THRESHOLD,TOP_THRESHOLD
from utils.geometry_utils import find_board_center_and_windows,Gaze_angle,FRAME_RATE
from kalman_filter import triangulate_kalman,DISTRUSTNESS,CUTOFF,dt
from utils.audio_processing import read_audio,sample_rate
from reproject_3d_to_2d import reproject_3d_to_2d
import os
import numpy as np
import warnings
import scipy.io.wavfile as wavfile
import shutil
import toml
import time
from utils.calibration_3d_utils import get_extrinsics
from utils.reorganize_utils import reorganize_mat_file
from utils.triangulation_utils import THRESHOLD as TRIANGULATE_THRESHOLD
from utils.squeaks_utils import Squeaks
import filecmp


# top camera dlc config
top_config = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\DLC\Alec_second_try-Devon-2020-12-07\config.yaml'
# side camera dlc config
#side_config = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\DLC\side_cameras-Devon-2021-03-10\config.yaml'
side_config = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\DLC\side_cameras_distorted-Devon-2021-03-17\config.yaml'
dlc_path = [top_config,side_config]
HDD_path = r'E:\behavior_data_archive'
server_path=r'R:\SchwartzLab\Behavior'


class ProcessingGroup:

	def __init__(self):
		self.rootpath = None
		self.dlcpath = None
		self.global_config_path = global_config_path
		self.in_calib = Calib('intrinsic')
		self.al_calib = Calib('alignment')
		self.ex_calib = Calib('extrinsic')
		self.server_thread=None
		self.squeak_ob=Squeaks()

	# rootpath refers to the working directory
	# dlc path is a default path for the trained dlc model
	def __call__(self, rootpath, dlcpath=None):
		if dlcpath is None:
			dlcpath = dlc_path
		self.dlcpath = dlcpath
		self.rootpath = rootpath
		self.processpath = 'undistorted'
		self.config_path = 'config'
		self.squeak_ob.set_root_path(rootpath)

	@property
	def processpath(self):
		return self._processpath

	@processpath.setter
	def processpath(self,folder_name):
		path = os.path.join(self.rootpath, folder_name)
		if not os.path.exists(path):
			os.mkdir(path)
		self._processpath=path

	@property
	def config_path(self):
		return self._config_path

	@config_path.setter
	def config_path(self, folder_name):
		path = os.path.join(self.rootpath, folder_name)
		#if not os.path.exists(path):
		#	os.mkdir(path)
		self._config_path = path

	def copy_configs(self):
		#TODO: save a reference to the config for each acquisition, then copy the right one, rather than the most recent one
		if self.rootpath:
			# if version == None it will copy the latest version
			version = copy_config(self.rootpath,version=None)
			return version
		else:
			raise Exception('root path incorrect/unimplemented')

	def check_process(self):
		# check under rootpath first
		# check calib:
		## is_done_intrinsic = self.check_intrinsic()
		## is done_alignment = self.check_alignment()
		## is done_extrinsic =self.check_extrinsic()
		## is_done_undistort =self.check_undistort()
		# check copied
		# check dlc
		# check dsqk
		# check uploaded to server
		# don't need to check if saved in HDD since the temp folder will be deleted
		# return {'intrinsic':is_done_intrinsic,
		# 		'alignment': is_done_alignment,
		# 		'extrinsic': is_done_extrinsic,
		# 		'undistorted: 'is_done_undistort,
		#		'copied': is copied,
		# 		'dlc': is_done_dlc,
		# 		'dsqk': is_done_dsqk,
		# 		'2server': is_done_2server	}
		pass

	def post_process(self,
					 intrinsic=True,
					 alignment=True,
					 extrinsic=True,
					 undistort=True,
					 copy_config=True,
					 dlc=True,
					 triangulate=True,
					 reproject=True,
					 gaze=True,
					 dsqk=True,
					 reorganize=True,
					 server=True,
					 HDD=True):

		# self.copy_configs()
		undistort_thread=None
		dlc_threads=None
		reproject_threads=None
		intrinsic_count=0
		alignment_count=0
		extrinsic_count=0

		if intrinsic:
			# do the following under behavior_rig/config/
			intrinsic_count = self.get_intrinsic_config()

		if alignment:
			alignment_count = self.get_alignment_config()
			self.find_windows()

		if extrinsic:
			extrinsic_count = self.get_extrinsic_config()

		# if any of the calibration happened: creat another version and save it to config_archive
		if intrinsic_count+alignment_count+extrinsic_count:
			archive_items=os.listdir(global_config_archive_path)
			if len(archive_items):
				versions=[int(re.findall(r"_v(\d+)_",i)[0]) for i in archive_items]
				latest=max(versions)
			else:
				latest=0
			date_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
			new_config_name=f"{date_time}_v{latest+1}_config"
			new_config_archive_path = os.path.join(global_config_archive_path,new_config_name)

			# archive to config_archive
			shutil.copytree(global_config_path,new_config_archive_path)

		if copy_config:

			version=self.copy_configs()
			self.add_local_config(version)

		if undistort:
			undistort_thread=threading.Thread(target=undistort_videos,args=(self.rootpath,))
			undistort_thread.start()
			#undistort_videos(self.rootpath)
		if dlc:
			dlc_threads=self.dlc_analysis()

		if gaze:
			self.gaze_analysis()
		if triangulate:
			triangulate_kalman(self.rootpath)
		if reproject:
			reproject_threads=reproject_3d_to_2d(self.rootpath)

		# join all the threads
		try:
			undistort_thread.join()
		except: pass

		try:
			for thread in dlc_threads:
				thread.join()
		except: pass

		try:
			for thread in reproject_threads:
				thread.join()
		except: pass

		if dsqk:
			self.audio_processing()
			self.dsqk_analysis(save_spectrogram=True)

		if reorganize:
			self.reorganize()
		if server:
			self.SSD2server_update()
			'''
			if self.server_thread is None:
				self.server_thread=threading.Thread(target=self.SSD2server)
				self.server_thread.start()
			else:
				self.server_thread.join()
				self.server_thread = threading.Thread(target=self.SSD2server)
				self.server_thread.start()
			'''
		if HDD:
			self.SSD2HDD()

	def add_local_config(self, version):
		local_config={'date':time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
					  'frame_rate':FRAME_RATE,
					  'sample_rate':sample_rate,
					  'dlc':
						   {'top_camera_threshold':TOP_THRESHOLD,
					  		'side_camera_threshold':SIDE_THRESHOLD},
					 'kalman_filter':{'dt':dt,
									  'distrustness':DISTRUSTNESS,
									  'cutoff':CUTOFF},
					  'triangulation':{'threshold':TRIANGULATE_THRESHOLD},
					  'config_version':version,

					  }
		filepath=os.path.join(self.config_path,'config_local.toml')
		with open(filepath,'w') as f:
			toml.dump(local_config,f)

	def get_intrinsic_config(self):
		# get item list
		# if count == 0 it means no calibration done
		count=0
		items = os.listdir(self.global_config_path)
		for item in items:
			if 'intrinsic' in item and 'temp' in item:
				temp_config_path = os.path.join(self.global_config_path, item)
				self.in_calib.save_processed_config(temp_config_path)
				try:
					os.remove(temp_config_path)
					count += 1
				except:
					Warning("can't remove the temp config file")
		return count

	def get_alignment_config(self):
		alignment = 'config_alignment_%s_temp.toml' % TOP_CAM
		path = os.path.join(self.global_config_path, alignment)
		self.al_calib.save_processed_config(path)
		try:
			os.remove(path)
			count=1
		except:
			Warning("can't remove the temp config file")
			count =0

		return count

	def get_extrinsic_config(self):
		# get item list
		items = os.listdir(self.global_config_path)
		intrinsic_list = []
		video_list = []
		board = self.ex_calib.board
		count=0

		for item in items:
			if 'intrinsic' in item and 'temp' not in item:
				intrinsic_path = os.path.join(self.global_config_path, item)
				intrinsic_list.append(intrinsic_path)
			if 'MOV' in item:
				video_path = os.path.join(self.global_config_path, item)
				video_list.append(video_path)

		if len(video_list)==4:

			# get intrinsic matrices and video list
			intrinsic_list.sort()
			video_list.sort()
			loaded = [toml.load(path) for path in intrinsic_list]
			has_top = [True if TOP_CAM in item else False for item in video_list]
			cam_align = has_top.index(True)
			vid_indices = list(range(len(has_top)))

			# calibration
			extrinsic, error = get_extrinsics(vid_indices,
											  video_list,
											  loaded,
											  cam_align,
											  board)
			time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
			extrinsic_new = dict(zip(map(str, extrinsic.keys()), extrinsic.values()))
			results = {'extrinsic': extrinsic_new,
					   'error': error,
					   'date': time_stamp}
			extrinsic_path = 'config_extrinsic.toml'
			saving_path = os.path.join(self.global_config_path,extrinsic_path)
			with open(saving_path, 'w') as f:
				toml.dump(results, f)

			count+=1

			# remove everything after calibration
			for item in items:
				if 'extrinsic' in item and 'temp' in item:
					os.remove(os.path.join(self.global_config_path,item))
		return count

	def find_windows(self):
		alignment = 'config_alignment_%s.toml'%TOP_CAM
		rig = 'config_behavior_rig.toml'

		with open(os.path.join(self.global_config_path, alignment), 'r') as f:
			config = toml.load(f)
		if 'recorded_center' not in config.keys():
			try:
				new_corners = np.array(config['undistorted_corners'])
			except:
				new_corners = np.array(config['corners'])
			ids = config['ids']
			ids = [int(i[0][0]) for i in ids]
			new_corners = np.array([corner for _, corner in sorted(zip(ids, new_corners))])

			if new_corners.shape != (6, 1, 4, 2):
				raise Exception("can't proceed! missing corners")
			with open(os.path.join(self.global_config_path, rig), 'r') as f:
				rig = toml.load(f)
			results = find_board_center_and_windows(new_corners, rig)
			with open(os.path.join(self.global_config_path, alignment), 'a') as f:
				toml.dump(results, f, encoder=toml.TomlNumpyEncoder())

	def dlc_analysis(self):
		# dlc anlysis on both top and side cameras
		threads=dlc_analysis(self.rootpath, self.dlcpath)
		if threads is not None:
			return threads
		else:
			return None

	def gaze_analysis(self):
		# gaze analysis on top camera
		warnings.filterwarnings('ignore') # ignore warning
		gaze_model = Gaze_angle(self.config_path)

		# binocular gaze
		gaze_model.gazePoint = 0
		bino = gaze_model(self.rootpath, save=True)
		gaze_model.plot(bino, savepath=self.rootpath)

		# monocular gaze
		gaze_model.gazePoint = 0.5725
		mono = gaze_model(self.rootpath, save=True)
		gaze_model.plot(mono,savepath=self.rootpath)

	def audio_processing(self):
		BK_filepath=os.path.join(self.rootpath,'B&K_audio.tdms')
		dodo_filepath=os.path.join(self.rootpath, 'dodo_audio.tdms')
		BK_audio=read_audio(BK_filepath)
		dodo_audio=read_audio(os.path.join(dodo_filepath))

		wavfile.write(BK_filepath[:-4] + 'wav', int(sample_rate), BK_audio[0])
		wavfile.write(dodo_filepath[:-4] + 'wav', int(sample_rate), dodo_audio[0])
		print("saved audio file")

	def dsqk_analysis(self,save_spectrogram=False):
		fname1 = 'Dodo_audio.wav'
		fname2 = 'B&K_audio.wav'
		filepath1=os.path.join(self.rootpath,fname1)
		filepath2=os.path.join(self.rootpath,fname2)
		saving_filepath1=os.path.join(self.rootpath,'Dodo_audio_squeaks.mat')
		saving_filepath2 = os.path.join(self.rootpath, 'B&K_audio_squeaks.mat')

		if not os.path.exists(saving_filepath1):
			call_time1 = self.squeak_ob(fname1)
			self.squeak_ob.draw_Dodo_squeaks(self.rootpath, call_time1)
		if not os.path.exists(saving_filepath2):
			call_time2=self.squeak_ob(fname2)
			self.squeak_ob.draw_BK_squeaks(self.rootpath, call_time2)
		if save_spectrogram:
			working_path=os.path.join(self.rootpath,filepath2[:-4]+'_squeaks.mat')
			self.squeak_ob.draw_spectrogram(working_path,self.rootpath)


	def reorganize(self):
		# TODO: add everything to .mat file

		reorganize_mat_file(self.rootpath,self.squeak_ob.load_squeak_time_matlab)

		items = os.listdir(self.rootpath)
		if 'reproject' in str(items):
			if not os.path.exists(os.path.join(self.rootpath, 'reproject')):
				os.mkdir(os.path.join(self.rootpath,'reproject'))
			for item in items:
				if os.path.isfile(os.path.join(self.rootpath, item)):
					if 'reproject' in item:
						file=os.path.join(self.rootpath,item)
						file2=os.path.join(self.rootpath,'reproject',item)
						shutil.move(file,file2)

		if 'DLC' in str(items):
			if not os.path.exists(os.path.join(self.rootpath, 'DLC')):
				os.mkdir(os.path.join(self.rootpath, 'DLC'))
			for item in items:
				if os.path.isfile(os.path.join(self.rootpath, item)):
					if 'DLC' in item:
						file = os.path.join(self.rootpath, item)
						file2 = os.path.join(self.rootpath, 'DLC', item)
						shutil.move(file, file2)

		if '.MOV' in str(items):
			if not os.path.exists(os.path.join(self.rootpath, 'raw')):
				os.mkdir(os.path.join(self.rootpath, 'raw'))
			for item in items:
				if os.path.isfile(os.path.join(self.rootpath, item)):
					if '.MOV' in item:
						file = os.path.join(self.rootpath, item)
						file2 = os.path.join(self.rootpath, 'raw', item)
						shutil.move(file, file2)

		if 'audio' in str(items):
			if not os.path.exists(os.path.join(self.rootpath, 'audio')):
				os.mkdir(os.path.join(self.rootpath, 'audio'))
			for item in items:
				if os.path.isfile(os.path.join(self.rootpath, item)):
					if 'audio' in item or 'squeaks' in item or 'spectrogram' in item:
						file = os.path.join(self.rootpath, item)
						file2 = os.path.join(self.rootpath, 'audio', item)
						shutil.move(file, file2)

		print('finish reorganizing folder %s'%self.rootpath)

	def SSD2server_update(self):
		filename = os.path.split(self.rootpath)[-1]
		animal_ID =os.path.split(os.path.split(self.rootpath)[0])[-1]
		full_server_path = os.path.join(server_path, animal_ID,filename)

		if not os.path.exists(os.path.join(server_path,animal_ID)):
			self.SSD2server()
		elif not os.path.exists(full_server_path):
			try:
				print(f"trying to upload {self.rootpath} to the server")
				shutil.copytree(self.rootpath, full_server_path)
				print(f'finished uploading {self.rootpath} to the server.')
			except shutil.Error as e:
				for src, dst, msg in e.args[0]:
					print(dst, src, msg)
		else:
			items=os.listdir(self.rootpath)
			for i in items:
				full_server_i=os.path.join(full_server_path,i)
				full_local_i=os.path.join(self.rootpath,i)
				if os.path.isdir(full_local_i): # if it is a folder
					if not os.path.exists(full_server_i):
						print(f"this folder {i} doesn't exist on server! uploading now....")
						shutil.copytree(full_local_i,full_server_i)
					else:
						subitems = os.listdir(full_local_i)
						for sub_i in subitems:
							full_local_sub_i=os.path.join(full_local_i,sub_i)
							full_server_sub_i=os.path.join(full_server_i,sub_i)
							if not os.path.isfile(full_server_sub_i) or not filecmp.cmp(full_server_sub_i, full_local_sub_i):
								print(f"this file {sub_i} doesnt's exist or is different in the subfolder {animal_ID}_{filename}_{i}. Overwriting... ")
								shutil.copyfile(full_local_sub_i, full_server_sub_i)
				else:# if it is a file
					if not os.path.isfile(full_server_i) or not filecmp.cmp(full_server_i,full_local_i):
						print(f"this file {i} doesnt's exist or is different in the folder {animal_ID}_{filename}. Overwriting... ")

						shutil.copyfile(full_local_i,full_server_i)

	def SSD2server(self):
		#config_path_server = os.path.join(server_path, 'config')

		#items_ssd=os.listdir(global_config_archive_path)
		#items_server = os.listdir(config_path_server)
		#if len(items_server)!=len(items_ssd):
		#	shutil.rmtree(config_path_server)
		#	shutil.copytree(global_config_archive_path,config_path_server)


		filename = os.path.split(self.rootpath)[-1]
		animal_ID =os.path.split(os.path.split(self.rootpath)[0])[-1]
		full_server_path = os.path.join(server_path, animal_ID,filename)

		if not os.path.exists(os.path.join(server_path,animal_ID)):
			os.mkdir(os.path.join(server_path,animal_ID))
		try:
			print("trying to upload to the server")
			shutil.copytree(self.rootpath, full_server_path)
			print('finished uploading to the server.')
		except shutil.Error as e:
			for src, dst, msg in e.args[0]:
				# src is source name
				# dst is destination name
				# msg is error message from exception
				print(dst, src, msg)


	def SSD2HDD(self):
		# first run double check
		# copy and paste
		name = os.path.split(self.rootpath)[-1]
		backup_path = os.path.join(HDD_path, name)
		try:
			shutil.copytree(self.rootpath, backup_path)
		except shutil.Error as e:
			for src, dst, msg in e.args[0]:
				# src is source name
				# dst is destination name
				# msg is error message from exception
				print(dst, src, msg)
		pass

	def full_revert(self,root_path):
		# move everything out+delete gaze analysis+delete undistort + delete DLC and config +delete audio+full_data.mat
		items=os.listdir(root_path)
		if 'raw' in items:
			subpath=os.path.join(root_path,'raw')
			subitems = os.listdir(subpath)
			for subitem in subitems:
				subitem_path=os.path.join(subpath,subitem)
				move2path=os.path.join(root_path,subitem)
				shutil.move(subitem_path, move2path)
			shutil.rmtree(subpath)
		if 'DLC' in items:
			subpath = os.path.join(root_path, 'DLC')
			shutil.rmtree(subpath)
		if 'gaze' in items:
			subpath = os.path.join(root_path, 'gaze')
			shutil.rmtree(subpath)
		if 'undistorted' in items:
			subpath = os.path.join(root_path, 'undistorted')
			shutil.rmtree(subpath)
		if 'config' in items:
			subpath= os.path.join(root_path, 'config')
			shutil.rmtree(subpath)
		if 'full_data.mat' in items:
			subpath = os.path.join(root_path, 'full_data.mat')
			os.remove(subpath)
			print('removed full data')

	def half_revert(self,root_path):
		# move everything out+delete gaze analysis+delete undistort+keep DLC and config
		items=os.listdir(root_path)
		if 'raw' in items:
			subpath=os.path.join(root_path,'raw')
			if os.path.exists(subpath):
				subitems = os.listdir(subpath)
				for subitem in subitems:
					subitem_path=os.path.join(subpath,subitem)
					move2path=os.path.join(root_path,subitem)
					shutil.move(subitem_path, move2path)
				shutil.rmtree(subpath)
		if 'DLC' in items:
			subpath = os.path.join(root_path, 'DLC')
			if os.path.exists(subpath):
				subitems = os.listdir(subpath)
				for subitem in subitems:
					subitem_path = os.path.join(subpath, subitem)
					move2path = os.path.join(root_path, subitem)
					shutil.move(subitem_path, move2path)
				shutil.rmtree(subpath)
		if 'gaze' in items:
			subpath = os.path.join(root_path, 'gaze')
			if os.path.exists(subpath):
				shutil.rmtree(subpath)
		if 'undistorted' in items:
			subpath = os.path.join(root_path, 'undistorted')
			if os.path.exists(subpath):
				shutil.rmtree(subpath)
		if 'audio' in items:
			subpath = os.path.join(root_path, 'audio')
			if os.path.exists(subpath):
				subitems = os.listdir(subpath)
				for subitem in subitems:
					subitem_path = os.path.join(subpath, subitem)
					move2path = os.path.join(root_path, subitem)
					shutil.move(subitem_path, move2path)
				shutil.rmtree(subpath)


if __name__ == '__main__':
	import tqdm
	# GLOBAL_CONFIG_PATH = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig'
	working_dir = r'D:\Desktop'
	items =os.listdir(working_dir)
	pg = ProcessingGroup()

	'''
	mouse='2160'
	item='2021-06-17_habituation_unnamedTrial'
	path = os.path.join(working_dir,mouse,item)
	pg(path)
	pg.post_process(intrinsic=False,
					alignment=True,
					extrinsic=False,
					undistort=False,
					copy=True,
					dlc=False,
					triangulate=False,
					reproject=False,
					gaze=True,
					dsqk=False,
					reorganize=True,
					server=False,
					HDD=False)
	'''
	server_items=os.listdir(server_path)

	useful=list(map(lambda x:x.isdigit(),items))
	new_items=list(np.array(items)[useful])
	for item in tqdm.tqdm(new_items):
		if int(item)>=1131:
			print(item)
			path = os.path.join(working_dir, item)
			server_animal_path = os.path.join(server_path, item)
			server_subitem=None

			subitems= os.listdir(path)
			subitems=list(sorted(subitems))
			for i, subitem in enumerate(subitems):
				try:
					server_subitem = os.listdir(server_animal_path)
				except:
					pass
				#if server_subitem is None or subitem not in server_subitem:

				if '07-23' in subitem or '07-22' in subitem:
					full_path=os.path.join(path,subitem)
					if full_path!='D:\\Desktop\\1131\\2021-06-17_habituation' and full_path!='D:\\Desktop\\1136\\2021-06-17_habituation':
							#pg.half_revert(full_path)
							'''
							print(subitem)
							local_config_path=os.path.join(full_path,'config')
							local_gaze_path = os.path.join(full_path, 'gaze')
							local_reorg_dlc_path=os.path.join(full_path,'DLC')
							local_full_data_path = os.path.join(full_path,'full_data.mat')
							remote_full_data_path = os.path.join(server_animal_path,subitem,'full_data.mat')
	
							gaze=Gaze_angle(local_config_path)
							gaze.gazePoint=0
							bino=gaze(local_reorg_dlc_path,save=False)
							#bino_path=os.path.join(full_path,'gaze_angle_0.mat')
							gaze.save(full_path,bino)
							print('saved gaze')
	
							gaze.gazePoint = 0.5725
							mono = gaze(local_reorg_dlc_path, save=False)
							#mono_path = os.path.join(full_path, 'gaze_angle_32.mat')
							gaze.save(full_path, mono)
							print('saved gaze')
	
							reorganize_mat_file2(full_path)
							#add_new_gaze_to_full_mat(local_gaze_path,local_full_data_path)
							shutil.copy(local_full_data_path,remote_full_data_path)
							'''
							pg(full_path)
							pg.post_process(intrinsic=False,
											alignment=False,
											extrinsic=False,
											undistort=False,
											copy_config=False,
											dlc=False,
											triangulate=False,
											reproject=False,
											reorganize=False,
											gaze=True,
											dsqk=True,
											server=False,
											HDD=False)



