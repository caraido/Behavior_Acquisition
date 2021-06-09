import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def get_speed(pose, arena_center, r):
	'''
	get_speed() takes the pose of each frame, arena center and the radius of the arena to calculate the speed of each time point
	'''
	left_ear = np.stack((
		pose['leftear']['x'],
		pose['leftear']['y'],
	)).transpose()  # should be t-by-2
	right_ear = np.stack((
		pose['rightear']['x'],
		pose['rightear']['y'],
	)).transpose()  # should be t-by-2

	head_center = (left_ear + right_ear) / 2
	in_donut = (head_center - arena_center) * 2 - r * 2
	in_experiment = in_donut.copy()
	in_experiment[in_donut < 0] = 0
	in_experiment[in_donut > 0] = 1

	speed = np.linalg.norm(np.diff(head_center))
	return speed


def get_smoothed_speed(speed, param,smoothing_method='moving_average'):
	if smoothing_method=='moving_average':
		# generic moving average smoothing
		return np.convolve(speed,np.ones((param,))/param,mode='full')
	elif smoothing_method=='Savitzky_Golay':
		# second roder savitzky golay smoothing
		return scipy.signal.savgol_filter(speed, param,2)
	else:
		raise ValueError('wrong/unimplemented smoothing method!')

def draw_speed(speed):
	fig=plt.figure()
	# TODO: implement this when testing

