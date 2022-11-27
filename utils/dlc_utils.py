import deeplabcut
import os
import threading
from global_settings import number_of_camera,TOP_CAM


def is_fully_analyzed(path):
	items=os.listdir(path)
	h5=[i for i in items if '.h5' in i]
	if len(h5)==number_of_camera:
		return True
	else:
		return False


def dlc_analysis(in_path, out_path, dlc_config_path,dlc_top,dlc_side):

	top_config = dlc_config_path[0]
	side_config = dlc_config_path[1]
	things = os.listdir(in_path)

	top = [a for a in things if '.MOV' in a and TOP_CAM in a]
	# top_path = [os.path.join(processed_path, top[i]) for i in range(len(top))]
	top_path = [os.path.join(in_path, top[i]) for i in range(len(top))]
	side = [a for a in things if '.MOV' in a and TOP_CAM not in a]
	# side_path = [os.path.join(processed_path, side[i]) for i in range(len(side))]
	side_path = [os.path.join(in_path, side[i]) for i in range(len(side))]

	# top camera
	if dlc_top:
		deeplabcut.analyze_videos(top_config,
								  top_path,
								  save_as_csv=True,
								  videotype='mov',
								  shuffle=1,
								  gputouse=0,
								  destfolder=out_path)

		deeplabcut.convert_detections2tracklets(top_config, top_path, videotype='mov', track_method='ellipse', destfolder=out_path)
		deeplabcut.stitch_tracklets(top_config, top_path, videotype='mov', track_method='ellipse', destfolder=out_path)
		deeplabcut.filterpredictions(top_config,top_path, track_method='ellipse', destfolder=out_path)

		arguments={'config':top_config,
					'videos':top_path,
					'save_frames':False,
					'trailpoints':1,
					'videotype':'mov',
					'filtered':True,
					"draw_skeleton":'True',
					"destfolder":out_path}
		create_labeled_video_top_thread=threading.Thread(target=deeplabcut.create_labeled_video,kwargs=arguments)
		create_labeled_video_top_thread.start()
	else:
		create_labeled_video_top_thread=None


	# side cameras
	if dlc_side:
		deeplabcut.analyze_videos(side_config,
								  side_path,
								  save_as_csv=True,
								  videotype='mov',
								  shuffle=1,
								  gputouse=0,
								  destfolder=out_path
								  )

		deeplabcut.convert_detections2tracklets(side_config, side_path, videotype='mov', track_method='ellipse', destfolder=out_path)
		deeplabcut.stitch_tracklets(side_config, side_path, videotype='mov', track_method='ellipse', destfolder=out_path)
		deeplabcut.filterpredictions(side_config,side_path, track_method='ellipse', destfolder=out_path)


		arguments={'config':side_config,
					'videos':side_path,
					'save_frames':False,
					'trailpoints':1,
					'videotype':'mov',
					"draw_skeleton":'True',
					"destfolder":out_path}
		create_labeled_video_side_thread = threading.Thread(target=deeplabcut.create_labeled_video, kwargs=arguments)
		create_labeled_video_side_thread.start()
	else:
		create_labeled_video_side_thread=None


	return create_labeled_video_top_thread,create_labeled_video_side_thread



