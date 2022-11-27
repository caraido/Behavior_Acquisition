import threading

import numpy as np
import cv2
import toml
import pandas as pd
import os
from matplotlib import cm as colormap

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


def project_points(xyz,extrinsic,intrinsic,dist_coeff,is_fisheye):  # xyz.shape == (-1, 4, 3)
    xy = np.empty((xyz.shape[0], xyz.shape[1], 2, 4))  # 4 markers, 2 dims, 4 cameras
    for k in range(xyz.shape[1]):
        xyzh = np.hstack((xyz[:, k, :], np.ones((xyz.shape[0], 1))))
        for j, (e, i, d) in enumerate(zip(extrinsic, intrinsic, dist_coeff)):
            imat = np.concatenate((i, np.array([0, 0, 1])[np.newaxis, :]), axis=0)
            proj = np.dot(imat, e)
            xyh = np.dot(proj, xyzh.T).T
            z = xyh[:, -1]
            rvec, _ = cv2.Rodrigues(e[0:3, 0:3])
            if j == 1:
                pos, _ = cv2.projectPoints(xyz[:, k, np.newaxis, :], rvec, e[0:3, 3], imat, d)
            else:
                pos, _ = cv2.fisheye.projectPoints(xyz[:, k, np.newaxis, :], rvec, e[0:3, 3], imat, d)
            if np.any(z < 0):
                pos[z < 0, :, :] = np.nan  # any points behind the camera should be discarded
            xy[:, k, :, j] = pos.squeeze()
    return xy


def xyz_to_xy(xyz, E, K, dist_coeff):
    xy_h = cv2.transform(xyz, K @ E) #homogeneous xy coordinates
    
    rvec, _ = cv2.Rodrigues(E[:3, :3])

    if len(dist_coeff) == 4:
        projectPoints = cv2.fisheye.projectPoints
    else:
        projectPoints = cv2.projectPoints
    
    xy,_ = projectPoints(xyz, rvec, E[:3,3], K, dist_coeff)
    xy[xy_h[:,:,2]<=0,:] = np.nan #these points are behind the camera
    
    return xy



def draw_markers(vid_path, output_path, cam_xy: pd.DataFrame):
    cap = cv2.VideoCapture(vid_path)
    vid_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    framesize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=framesize, isColor=False)
    csv_length = len(cam_xy)
    length = min(vid_length, csv_length)

    # col_length = int(cam_xy.shape[1] / 2)
    n_markers = len(cam_xy.columns.get_level_values(0).unique())

    cm = colormap.get_cmap('rainbow', n_markers)
    colors = [[int(c*255) for c in cm(i)] for i in range(n_markers)]

    ret = True
    count = 0

    while ret and count < length:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # iterate body part
        for i,c in enumerate(cam_xy.columns.get_level_values(0).unique()):
            # index the columns with i
            # index the row with count
            x,y = cam_xy.iloc[count][c]
            if x > 0 and x < framesize[0] and y < framesize[1] and y > 0:
                frame = cv2.circle(frame, (int(x),int(y)), radius=8, color=colors[i], thickness=-1)

        # write the frame
        writer.write(frame)
        count += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    return


def reproject_3d_to_2d(raw_path, tracked_path, config_path, reprojected_path):
    # get camera matrices
    cam_mat, intrinsic, extrinsic, dist_coeff = get_cam_mat(config_path)
    
    pose_3d = pd.read_csv(os.path.join(tracked_path, 'output_3d_data_kalman.csv'), header=[0,1])
    marker_names = pose_3d.columns.get_level_values(0).unique()

    xyz = pose_3d.drop(['v_x','v_y','v_z'],level=1,axis=1).values.reshape(-1,1,3)
    cam = []
    for i,(e,k,d) in enumerate(zip(extrinsic, intrinsic, dist_coeff)):
        #TODO: ought to save these by serial number, for consistency...

        uv = xyz_to_xy(xyz, e, k, d)

        cam.append(pd.DataFrame(uv.reshape(len(pose_3d), -1), columns = pd.MultiIndex.from_product((marker_names , ('x','y')))))        
        cam[-1].to_csv(os.path.join(reprojected_path, f'reproject_cam{i}_xy.csv'))

    # draw markers to the video
    threads=[]
    items = os.listdir(raw_path)
    vid_list = sorted([a for a in items if '.MOV' in a])
    for j, vid in enumerate(vid_list):
        vid_path = os.path.join(raw_path, vid)
        saving_path = os.path.join(reprojected_path, 'reprojected_' + vid)

        # starting threads
        thread=threading.Thread(target=draw_markers,args=(vid_path,saving_path,cam[j],))
        thread.start()
        threads.append(thread)

        #draw_markers(vid_path, saving_path, cam[j])
        print("drew reprojected markers on the video %d"%j)

    return threads

if __name__ == '__main__':
    working_path = r'/Users/tianhaolei/Downloads/pose-3'
    # get camera matrices
    cam_mat, intrinsic, extrinsic, dist_coeff = get_cam_mat(working_path)
    # manually change the extrinsic matrix TODO
    # extrinsic[1][0:3, 3] = np.array([-.9, -1.6, -.55351041])

    # define body parts,2d,3d points column
    body_parts = ['left_ear', 'right_ear', 'nose', 'tail_base']
    xy_columns = ['left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y', 'nose_x', 'nose_y', 'tail_base_x',
                  'tail_base_y']
    xyz_columns = ['left_ear_x', 'left_ear_y', 'left_ear_z',
                   'right_ear_x', 'right_ear_y', 'right_ear_z',
                   'nose_x', 'nose_y', 'nose_z',
                   'tail_base_x', 'tail_base_y', 'tail_base_z']

    # directly read csv (from anipose result)
    pose_3d = pd.read_csv(os.path.join(working_path, 'output_3d_data.csv'))
    length = len(pose_3d)

    # make Dataframe from npy files (from kalman filter)
    marker0 = np.load(os.path.join(working_path, 'marker0.npy'))[:, 0:3]
    marker1 = np.load(os.path.join(working_path, 'marker1.npy'))[:, 0:3]
    marker2 = np.load(os.path.join(working_path, 'marker2.npy'))[:, 0:3]
    marker3 = np.load(os.path.join(working_path, 'marker3.npy'))[:, 0:3]
    markers = [marker0, marker1, marker2, marker3]

    # comment this part out if using anipose result
    '''
	pose_3d=pd.DataFrame(columns=xyz_columns,index=pd.Index(range(len(marker0))))
	for part,marker in zip(body_parts,markers):
	x=part+'_x'
	y=part+'_y'
	z=part+'_z'
	pose_3d[[x,y,z]]=marker
	'''

    # for each camera:
    cam0_xy = pd.DataFrame(columns=xy_columns)
    cam1_xy = pd.DataFrame(columns=xy_columns)
    cam2_xy = pd.DataFrame(columns=xy_columns)
    cam3_xy = pd.DataFrame(columns=xy_columns)

    # 3d to 2d
    for part in body_parts:
        x = part + '_x'
        y = part + '_y'
        z = part + '_z'

        xyz = np.array([pose_3d[x].values, pose_3d[y].values, pose_3d[z].values]).astype('float64')
        xy0 = xyz_to_xy(xyz, extrinsic[0], intrinsic[0], dist_coeff[0], True)
        xy1 = xyz_to_xy(xyz, extrinsic[1], intrinsic[1], dist_coeff[1], False)
        xy2 = xyz_to_xy(xyz, extrinsic[2], intrinsic[2], dist_coeff[2], True)
        xy3 = xyz_to_xy(xyz, extrinsic[3], intrinsic[3], dist_coeff[3], True)

        cam0_xy[x] = xy0[:, 0, 0]
        cam0_xy[y] = xy0[:, 0, 1]

        cam1_xy[x] = xy1[:, 0, 0]
        cam1_xy[y] = xy1[:, 0, 1]

        cam2_xy[x] = xy2[:, 0, 0]
        cam2_xy[y] = xy2[:, 0, 1]

        cam3_xy[x] = xy3[:, 0, 0]
        cam3_xy[y] = xy3[:, 0, 1]

    # save the results
    cam0_xy.to_csv(os.path.join(working_path, 'cam0_xy.csv'))
    cam1_xy.to_csv(os.path.join(working_path, 'cam1_xy.csv'))
    cam2_xy.to_csv(os.path.join(working_path, 'cam2_xy.csv'))
    cam3_xy.to_csv(os.path.join(working_path, 'cam3_xy.csv'))
    cam = [cam0_xy, cam1_xy, cam2_xy, cam3_xy]

    # draw markers to the video
    items = os.listdir(working_path)
    vid_list = sorted([a for a in items if '.MOV' in a])
    for j, vid in enumerate(vid_list):
        vid_path = os.path.join(working_path, vid)
        saving_path = os.path.join(working_path, 'reprojected_' + vid)
        draw_markers(vid_path, saving_path, cam[j])
