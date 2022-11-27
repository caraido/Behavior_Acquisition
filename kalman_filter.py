import numpy as np
import cv2
import toml
import pandas as pd
import os
from global_settings import dt,CUTOFF,DISTRUSTNESS



def get_rid_of_outliers(points):
	# no point can be out of boundary
	cond1 = points[:, :, 0] < 0
	cond2 = points[:, :, 0] > 1280
	cond3 = points[:, :, 1] < 0
	cond4 = points[:, :, 1] > 1024
	points[cond1, :] = np.nan
	points[cond2, :] = np.nan
	points[cond3, :] = np.nan
	points[cond4, :] = np.nan

	# points[:] = np.nan  # this one set all the points to np.nan
	return points


def undistort_points(all_points_raw, in_mat, dist_coeff, fisheyes: list):
	all_points_und = np.zeros(all_points_raw.shape)
	if int(len(all_points_raw) / 2) != len(fisheyes):
		raise Exception
	else:
		for i in range(len(in_mat)):
			points_2d = all_points_raw[2 * i:2 * i + 2, np.newaxis].swapaxes(0, 2)
			if fisheyes[i]:
				Knew = in_mat[i].copy()
				Knew[(0, 1), (0, 1)] = 1 * Knew[(0, 1), (0, 1)]
				points_new = cv2.fisheye.undistortPoints(points_2d.astype('float64'),
														 in_mat[i].astype('float64'),
														 dist_coeff[i].astype('float64'), P=Knew.astype(
						'float64'))  # reverting due to issues with filter

				points_new = get_rid_of_outliers(points_new)
			else:
				# just for testing
				Knew = in_mat[i].copy()
				Knew[(0, 1), (0, 1)] = 1 * Knew[(0, 1), (0, 1)]
				# cv2.undistortPoints(points_2d.astype('float64'),
				points_new = cv2.undistortPoints(points_2d.astype('float64'),
												 in_mat[i].astype('float64'),
												 dist_coeff[i].astype('float64'), P=Knew.astype('float64'))
			all_points_und[2 * i:2 * i + 2] = points_new[:, 0, :].transpose()
	return all_points_und

def get_cam_mat(path):
	in_path = []
	ex_path = None
	items = sorted(os.listdir(path))
	for item in items:
		if '.toml' in item and 'extrinsic' in item and 'temp' not in item:
			ex_path = os.path.join(path, item)
		if '.toml' in item and 'intrinsic' in item and 'temp' not in item:
			in_path.append(os.path.join(path, item))

	# get camera mat
	in_mat = np.array(list(map(lambda x: np.array(
		toml.load(x)['camera_mat']).astype(float), in_path)))
	ex_data = toml.load(ex_path)
	ex_mat = np.array(list(ex_data['extrinsic'].values())).astype('float')
	ex_mat = ex_mat[:, :3, :]
	# only process the 2nd and 3rd dimension and leave 1st as index
	cam_mat = np.einsum("ijk,ikn->ijn", in_mat, ex_mat)
	dist_coeff = list(map(lambda x: np.array(
		toml.load(x)['dist_coeff']), in_path))

	return cam_mat, in_mat, ex_mat, dist_coeff

def get_pose(path, top_model, side_model):
	#Needs to only draw from the current models!
	top_model_name = top_model.split(os.sep)[-2].split('-')[0]
	side_model_name = side_model.split(os.sep)[-2].split('-')[0]
	

	dfs = []
	markers = []

	items = os.listdir(path)
	items = sorted(items)
	for item in items:
		if '.csv' in item and 'camera' in item and (top_model_name in item or side_model_name in item):
			full_path = os.path.join(path, item)
			data = pd.read_csv(full_path, header=[1, 2, 3]).center_mouse

			markers.append(data.columns.get_level_values(0).unique())
			for c in markers[-1]:
				data.loc[data.loc[:,(c,'likelihood')] < CUTOFF, (c,['x','y'])] = np.nan
			dfs.append(data)
			
	marker_set = set.intersection(*map(set, markers))
	min_length = min(len(df) for df in dfs)

	return pd.concat({i:df.loc[:min_length-1, marker_set] for i,df in enumerate(dfs)}, axis=1), marker_set
class Marker:
	DISTRUSTNESS = 1e22

	def __init__(self, pose, dt,config_path):
		self.length = pose.shape[1]
		self.x = np.empty((self.length, 6))  # *np.nan
		self.x[:] = np.nan
		self.xhat = None # TODO: do we want to index these?

		self.z = pose.transpose(0,2,1).reshape(-1,self.length)
		self.F = np.eye(6)
		self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt / 2
		self.P = np.empty((6, 6, self.length + 1))  # .fill(np.nan)# 1/3*np.eye(6)
		self.P[:] = np.nan
		self.P[:, :, 0] = np.eye(6) * 1 / 3
		#         self.P = np.eye(6)*1/3
		self.Phat = np.empty((6, 6, self.length))  # .fill(np.nan)
		self.Phat[:] = np.nan
		self.Q = dt * np.eye(6)
		self.R = dt * np.eye(8)
		# self.K = np.zeros((6, 8))
		self.adj_vector = np.empty(8)
		self.H = np.zeros((8, 6))
		self.i = -1
		self.read_config(config_path)

	def read_config(self, path):
		self.C, I, self.E, self.dist = get_cam_mat(path)

		self.rot = self.E[:, :, 0:3]

		# self.E[1][0:3, 3] = np.array([-.9, -1.6, -.55351041])  # manual correction
		self.trans = self.E[:, :, 3][:, :, np.newaxis]
		self.I = I[:, 0:2, :]
		self.IR = np.einsum('ijk, ikl->ijl', self.I, self.rot)

	def estimate_z_hat(self, x_hat):
		z_hat = np.einsum('ijk,kl->ijl', self.rot, x_hat[0:3, :]) + self.trans
		m = 1 / z_hat[:, 2]
		return np.einsum('ijk,ikl,il->ijl', self.I, z_hat, m).reshape((8, -1))

	def project_points(self, xyz):  # xyz.shape == (-1, 4, 3)

		xy = np.empty((xyz.shape[0], xyz.shape[1], 2, 4))  # 4 markers, 2 dims, 4 cameras
		for k in range(xyz.shape[1]):
			#         xyz =  state.markers[k].x[:,np.newaxis,:3].astype('float64') #N-by-1-by-3
			#     xyz = p3d[:,:,k].reshape(-1,1,3)#N*3*4
			xyzh = np.hstack((xyz[:, k, :], np.ones((xyz.shape[0], 1))))

			for j, (e, i, d) in enumerate(zip(self.E, self.I, self.dist)):
				imat = np.concatenate((i, np.array([0, 0, 1])[np.newaxis, :]), axis=0)
				proj = np.dot(imat, e)
				xyh = np.dot(proj, xyzh.T).T
				z = xyh[:, -1]
				rvec, _ = cv2.Rodrigues(e[0:3, 0:3])
				#             imat = np.concatenate((i,np.array([0,0,1])[np.newaxis,:]),axis=0)
				if j == 1:
					pos, _ = cv2.projectPoints(xyz[:, k, np.newaxis, :], rvec, e[0:3, 3], imat, d)
				else:
					pos, _ = cv2.fisheye.projectPoints(xyz[:, k, np.newaxis, :], rvec, e[0:3, 3], imat, d)
				if np.any(z < 0):
					#                 print(z.shape, pos.shape)
					pos[z < 0, :, :] = np.nan  # any points behind the camera should be discarded
				xy[:, k, :, j] = pos.squeeze()

		return xy

	def forward(self):
		#See Merven, B (2004). Person Tracking in 3D Using Kalman Filtering in Single and Multiple Camera Environments
		
		self.i += 1
		
		#project the model onto each camera [calculate z(t|x(t|t-1))]
		# NOTE: H is non-linear, so at each step we locally linearize it using the Jacobian
		# See https://stats.stackexchange.com/questions/497283/how-to-derive-camera-jacobian
		# If x(t|t-1) is a poor estimate, then the linearity assumption will fail 
		z_hat = np.einsum('ijk,kl->ijl', self.rot, self.x_hat[0:3, :]) + self.trans
		m = 1 / z_hat[:, 2]
		self.H[:, 0:3] = (np.einsum('ijk,il->ijk', self.IR, m) -
						  np.einsum('ijk,ikl,im,il,il->ijm', self.I, z_hat, self.rot[:, 2, :], m, m)).reshape((8, 3))
		z_hat_norm = np.einsum('ijk,ikl,il->ijl', self.I, z_hat, m).reshape((8,))

		#correct for untracked frames such that they don't contribute to the model
		n = np.isnan(self.z[:, self.i])
		self.z[n, self.i] = z_hat_norm[n]
		self.adj_vector[:] = 0
		self.adj_vector[n] = Marker.DISTRUSTNESS

		#estimate the relative strengths of the model and tracking noise [calculate K(t)]
		R_new = self.R + np.diag(self.adj_vector)
		HP = self.H @ self.Phat[:, :, self.i]
		inverse_mat = np.linalg.inv((HP @ self.H.T) + R_new)
		K = self.Phat[:, :, self.i] @ self.H.T @ inverse_mat

		#adjust the model using the tracks according to the Kalman gain K [calculate x(t|t), P(t|t)]
		self.P[:, :, self.i + 1] = self.Phat[:, :, self.i] - (K @ HP)
		self.x[self.i + 1, :] = (self.x_hat + (K @ (self.z[:, self.i] - z_hat_norm)[:, np.newaxis]))[:, 0]
		self.z[n, self.i] = np.nan

		#evolve the model to the next state state [calculate x(t+1|t), P(t+1|t)]
		self.Phat[:, :, self.i + 1] = (self.F @ self.P[:, :, self.i + 1] @ self.F.T) + self.Q
		self.x_hat = (self.F @ self.x[self.i + 1, :].T)[:, np.newaxis]


	def reverse(self):  # for bidirectional filter
		#See Yu, BM et al (2004). Derivation of Kalman Filtering and Smoothing Equations

		
		J = self.P[:, :, self.i] @ self.F.T @ np.linalg.inv(self.Phat[:, :, self.i])
		C = self.x[self.i + 1, :] - (self.F @ self.x[self.i, :])
		self.x[self.i, :] = self.x[self.i, :] + (J @ C)

		self.i -= 1
			
		
		# NOTE:
		#	J(t) = Var[x(t) | y(1...t)] @ F.T @ inv(Var[x(t+1) | y(1...t)])
		# 	C(t) = E[x(t+1) | y(1...T)] - F @ E[x(t) | y(1...t)]
		# 	E[x(t | y(1...T))] = E[x(t | y(1...t))] + J(t) @ C(t)
		#	
		#	E[x(t | y(1...t))] ~ self.x(t), result from forward filter
		#	Var[x(t) | y(1...t)] ~ self.P(t), result from forward filter
		#	Var[x(t+1) | y(1...t)] ~ self.Phat(t+1), result from forward filter
		#	E[x(t+1) | y(1...T)] ~ self.x(t+1), result from previous iteration of reverse filter
		
		#	for t = T,
		# 		E[x(t+1) | y(1...t)] = E[x(t+1) | y(1...T)]	
		#	and
		#		Var[x(t+1) | y(1...t)] = Var[x(t+1) | y(1...T)]
		#	so we can start at t = T-1
class Animal:
	def __init__(self, markers, dt,config_path):
		self.markers = [Marker(markers[:,:,i,:], dt,config_path) for i in range(markers.shape[2])]

	def locate(self, ind, location):
		marker = self.markers[ind]

		marker.x[0, :] = location
		marker.Phat[:, :, 0] = (marker.F @ marker.P[:, :, 0] @ marker.F.T) + marker.Q
		marker.x_hat = (marker.F @ marker.x[0, :].T)[:, np.newaxis]

	def forward(self, ind):
		self.markers[ind].forward()

	def reverse(self, ind):
		self.markers[ind].reverse()



def triangulate_kalman(tracked_path,config_path,top_model, side_model,causal = False):
	markers, marker_names=get_pose(tracked_path, top_model, side_model)
	C, I, E, dist = get_cam_mat(config_path)

	undistorted = np.empty((len(C), len(markers), len(marker_names), 2))
	for (i,cam),k,d in zip(markers.groupby(level=0, axis=1), I, dist):
		xy = cam.drop('likelihood', axis=1, level=2).values.reshape(len(markers),-1,2).reshape(-1,1,2)
		if len(d) == 4:
			undistortPoints = cv2.fisheye.undistortPoints
		else:
			undistortPoints = cv2.undistortPoints
		#NOTE: we want to preserve the camera matrix for use in the kalman filter
		undistorted[i,:,:,:] = get_rid_of_outliers(undistortPoints(xy, k, d, P=k)).reshape(len(markers),-1,2)

	state=Animal(undistorted,dt,config_path)
	for j in range(len(marker_names)):
		state.locate(j, np.random.randn(6) * .1 + np.array([0, 0, 31, 0, 0, 0]))
		for i in range(state.markers[j].length - 1):
			state.forward(j)
		if not causal:
			for i in range(state.markers[j].length - 1):
				state.reverse(j)

	pose_3d = pd.DataFrame(np.concatenate([m.x for m in state.markers], axis=1), columns = pd.MultiIndex.from_product((marker_names , ['x','y','z','v_x','v_y','v_z'])))

	save_path = os.path.join(tracked_path,'output_3d_data_kalman.csv')
	pose_3d.to_csv(save_path, index=False)


def main():
	rootdir = 'D:\\Desktop\\1218\\827_2021-09-22_female_unknown_social_status_female_2_males_(A)stranger_(B)stranger_(C)stranger\\'

	triangulate_kalman(os.path.join(rootdir, 'DLC'), os.path.join(rootdir, 'config'), causal=False)
	from reproject_3d_to_2d import reproject_3d_to_2d
	reproject_3d_to_2d(os.path.join(rootdir,'raw'), os.path.join(rootdir, 'DLC'), os.path.join(rootdir, 'config'), os.path.join(rootdir, 'reproject'))


if __name__ == "__main__":
	main()