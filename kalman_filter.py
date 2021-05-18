import numpy as np
import cv2
import toml
import pandas as pd
import os

dt = 1 / 30
CUTOFF = 0.7
DISTRUSTNESS = 1e22


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
		if '.toml' and 'extrinsic' in item:
			ex_path = os.path.join(path, item)
		if '.toml' and 'intrinsic' in item:
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


def get_pose(path):
	leftears = []
	rightears = []
	snouts = []
	tailbases = []

	items = os.listdir(path)
	items = sorted(items)
	for item in items:
		if '.csv' in item and 'camera' in item:
			full_path = os.path.join(path, item)
			data = pd.read_csv(full_path, header=[1, 2])
			data.loc[data.leftear.likelihood < CUTOFF, [
				('leftear', 'x'), ('leftear', 'y')]] = np.nan
			data.loc[data.rightear.likelihood < CUTOFF, [
				('rightear', 'x'), ('rightear', 'y')]] = np.nan
			data.loc[data.snout.likelihood < CUTOFF, [
				('snout', 'x'), ('snout', 'y')]] = np.nan
			data.loc[data.tailbase.likelihood < CUTOFF, [
				('tailbase', 'x'), ('tailbase', 'y')]] = np.nan

			leftear = np.array([data['leftear']['x'], data['leftear']['y']])
			rightear = np.array([data['rightear']['x'], data['rightear']['y']])
			snout = np.array([data['snout']['x'], data['snout']['y']])
			tailbase = np.array([data['tailbase']['x'], data['tailbase']['y']])

			leftears.append(leftear)
			rightears.append(rightear)
			snouts.append(snout)
			tailbases.append(tailbase)

			minlength = min([item.shape[1] for item in leftears])
			leftears = [item[:, :minlength] for item in leftears]
			rightears = [item[:, :minlength] for item in rightears]
			snouts = [item[:, :minlength] for item in snouts]
			tailbases = [item[:, :minlength] for item in tailbases]

	return np.array(leftears).reshape(-1, minlength), \
		   np.array(rightears).reshape(-1, minlength), \
		   np.array(snouts).reshape(-1, minlength), \
		   np.array(tailbases).reshape(-1, minlength)


class Marker:
	DISTRUSTNESS = 1e22

	def __init__(self, pose, dt,config_path):
		self.length = pose.shape[1]
		self.x = np.empty((self.length, 6))  # *np.nan
		self.x[:] = np.nan

		self.z = pose
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
		self.K = np.zeros((6, 8))
		self.adj_vector = np.empty(8)
		self.H = np.zeros((8, 6))
		self.i = 0
		self.read_config(config_path)

	def read_config(self, path):
		self.C, I, self.E, self.dist = get_cam_mat(path)

		self.rot = self.E[:, :, 0:3]

		self.E[1][0:3, 3] = np.array([-.9, -1.6, -.55351041])  # manual correction
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
		self.Phat[:, :, self.i] = np.dot(self.F, np.dot(self.P[:, :, self.i], self.F.T)) + self.Q
		#         P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

		x_hat = np.dot(self.F, self.x[self.i, :].T)[:, np.newaxis]
		z_hat = np.einsum('ijk,kl->ijl', self.rot, x_hat[0:3, :]) + self.trans
		m = 1 / z_hat[:, 2]
		self.H[:, 0:3] = (np.einsum('ijk,il->ijk', self.IR, m) -
						  np.einsum('ijk,ikl,im,il,il->ijm', self.I, z_hat, self.rot[:, 2, :], m, m)).reshape((8, 3))
		z_hat_norm = np.einsum('ijk,ikl,il->ijl', self.I, z_hat, m).reshape((8,))

		n = np.isnan(self.z[:, self.i])
		self.z[n, self.i] = z_hat_norm[n]
		self.adj_vector[:] = 0
		self.adj_vector[n] = Marker.DISTRUSTNESS
		R_new = self.R + np.diag(self.adj_vector)

		HP = np.dot(self.H, self.Phat[:, :, self.i])
		#         HP = np.dot(self.H, P)

		inverse_mat = np.linalg.inv(np.dot(HP, self.H.T) + R_new)
		self.K = np.dot(self.Phat[:, :, self.i], np.dot(self.H.T, inverse_mat))
		#         self.K = np.dot(P, np.dot(self.H.T, inverse_mat))

		self.P[:, :, self.i + 1] = self.Phat[:, :, self.i] - np.dot(self.K, HP)
		#         self.P = P - np.dot(self.K, HP)

		self.x[self.i + 1, :] = (x_hat + np.dot(self.K, (self.z[:, self.i] - z_hat_norm)[:, np.newaxis]))[:, 0]
		self.z[n, self.i] = np.nan

		self.i += 1

	def reverse(self):  # for bidirectional filter
		self.i -= 1
		J = np.dot(self.P[:, :, self.i + 1], np.dot(self.F.T, np.linalg.inv(self.Phat[:, :, self.i])))
		C = self.x[self.i + 1, :] - np.dot(self.F, self.x[self.i, :])
		self.x[self.i, :] = self.x[self.i, :] + np.dot(J, C)


class Animal:
	def __init__(self, markers, dt,config_path):
		self.markers = [Marker(marker, dt,config_path) for marker in markers]

	def locate(self, marker, location):
		self.markers[marker].x[0, :] = location

	def forward(self, marker):
		self.markers[marker].forward()

	def reverse(self, marker):
		self.markers[marker].reverse()



def triangulate_kalman(root_path):
	config_path=os.path.join(root_path,'config')
	markers=get_pose(root_path)
	C, I, E, dist = get_cam_mat(config_path)
	markers=[undistort_points(m, I, dist, fisheyes=[1,0,1,1]) for m in markers]
	state=Animal(markers,dt,config_path)
	for j in range(4):
		state.locate(j, np.random.randn(6) * .1 + np.array([0, 0, 31, 0, 0, 0]))
		for i in range(state.markers[j].length - 1):
			state.forward(j)
		for i in range(state.markers[j].length - 1):
			state.reverse(j)

	marker0 = state.markers[0].x[:, 0:3]
	marker1 = state.markers[1].x[:, 0:3]
	marker2 = state.markers[2].x[:, 0:3]
	marker3 = state.markers[3].x[:, 0:3]
	markers = [marker0, marker1, marker2, marker3]

	body_parts = ['leftear', 'rightear', 'snout', 'tailbase']
	xyz_columns = ['leftear_x', 'leftear_y', 'leftear_z',
				   'rightear_x', 'rightear_y', 'rightear_z',
				   'snout_x', 'snout_y', 'snout_z',
				   'tailbase_x', 'tailbase_y', 'tailbase_z']
	pose_3d = pd.DataFrame(columns=xyz_columns, index=pd.Index(range(len(marker0))))
	for part, marker in zip(body_parts, markers):
		x = part + '_x'
		y = part + '_y'
		z = part + '_z'
		pose_3d[[x, y, z]] = marker

	save_path = os.path.join(root_path,'output_3d_data_kalman.csv')
	pose_3d.to_csv(save_path)



