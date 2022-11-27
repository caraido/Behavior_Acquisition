import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def get_speed(pose):
	'''
	get_speed() takes the pose of each frame, arena center and the radius of the arena to calculate the speed of each time point
	'''
	left_ear = np.stack((
		pose['left_ear']['x'],
		pose['left_ear']['y'],
	)).transpose()  # should be t-by-2
	right_ear = np.stack((
		pose['right_ear']['x'],
		pose['right_ear']['y'],
	)).transpose()  # should be t-by-2

	head_center = (left_ear + right_ear) / 2

	speed = np.linalg.norm(np.diff(head_center,axis=0),axis=1)
	return speed


def get_smoothed_speed(speed, param,smoothing_method='median_filter'):
	if smoothing_method=='moving_average':
		# generic moving average smoothing
		return np.convolve(speed,np.ones((param,))/param,mode='full')
	elif smoothing_method=='Savitzky_Golay':
		# second order savitzky golay smoothing
		return scipy.signal.savgol_filter(speed, param,2)
	elif smoothing_method=='median_filter':
		return scipy.signal.medfilt(speed,param)
	else:
		raise ValueError('wrong/unimplemented smoothing method!')


