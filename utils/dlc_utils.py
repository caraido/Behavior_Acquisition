import deeplabcut
import os


def dlc_analysis(root_path, dlc_config_path):
	if isinstance(dlc_config_path, list):
		top_config = dlc_config_path[0]
		side_config = dlc_config_path[1]
		#processed_path = os.path.join(root_path, 'processed')
		#config_path = os.path.join(root_path, 'config')
		# things = os.listdir(processed_path)
		things = os.listdir(root_path)

		top = [a for a in things if '.MOV' in a and '17391304' in a]
		# top_path = [os.path.join(processed_path, top[i]) for i in range(len(top))]
		top_path = [os.path.join(root_path, top[i]) for i in range(len(top))]
		side = [a for a in things if '.MOV' in a and '17391304' not in a]
		# side_path = [os.path.join(processed_path, side[i]) for i in range(len(side))]
		side_path = [os.path.join(root_path, side[i]) for i in range(len(side))]

		# top camera
		deeplabcut.analyze_videos(top_config,
								  top_path,
								  save_as_csv=True,
								  videotype='mov',
								  shuffle=1,
								  gputouse=0)
		deeplabcut.create_labeled_video(top_config,
										top_path,
										save_frames=False,
										trailpoints=1,
										videotype='mov',
										draw_skeleton='True')
		# side cameras
		deeplabcut.analyze_videos(side_config,
								  side_path,
								  save_as_csv=True,
								  videotype='mov',
								  shuffle=1,
								  gputouse=0)
		deeplabcut.create_labeled_video(side_config,
										side_path,
										save_frames=False,
										trailpoints=1,
										videotype='mov',
										draw_skeleton='True')

