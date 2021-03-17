import pandas as pd
import numpy as np
from numpy import array as arr
import cv2

from tqdm import trange

THRESHOLD = 0.8

def reprojection_error(p3d, points2d, camera_mats):
    proj = np.dot(camera_mats, p3d)
    proj = proj[:, :2] / proj[:, 2, None]
    errors = np.linalg.norm(proj - points2d, axis=1)
    return np.mean(errors)


def triangulate_points(the_points, cam_mats):
    p3ds = []
    errors = []
    for ptnum in range(the_points.shape[0]):
        points = the_points[ptnum]
        good = ~np.isnan(points[:, 0])
        p3d = triangulate_simple(points[good], cam_mats[good])
        err = reprojection_error(p3d, points[good], cam_mats[good])
        p3ds.append(p3d)
        errors.append(err)
    p3ds = np.array(p3ds)
    errors = np.array(errors)
    return p3ds, errors

def triangulate_simple(points, camera_mats):
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d / p3d[3]
    return p3d


def distort_points_cams(points, camera_mats):
    out = []
    for i in range(len(points)):
        point = np.append(points[i], 1)
        mat = camera_mats[i]
        new = mat.dot(point)[:2]
        out.append(new)
    return np.array(out)


def reprojection_error_und(p3d, points2d, camera_mats, camera_mats_dist):
    proj = np.dot(camera_mats, p3d)
    proj = proj[:, :2] / proj[:, 2, None]
    proj_d = distort_points_cams(proj, camera_mats_dist)
    points2d_d = distort_points_cams(points2d, camera_mats_dist)
    errors = np.linalg.norm(proj_d - points2d_d, axis=1)
    return np.mean(errors)


def get_bp_interested(data: pd.DataFrame):
    header = data.keys()
    head = header.get_level_values(0)
    head = list(set(head))
    head.remove('bodyparts')
    return head


def read_single_2d_data(data: pd.DataFrame):
    length = len(data.index)
    index = arr(data.index)

    bp_interested = get_bp_interested(data)

    coords = np.zeros((length, len(bp_interested), 2))
    scores = np.zeros((length, len(bp_interested)))

    for bp_idx, bp in enumerate(bp_interested):
        bp_coords = arr(data[bp])
        coords[index, bp_idx, :] = bp_coords[:, :2]
        scores[index, bp_idx] = bp_coords[:, 2]

    return {'length': length,
            'coords': coords,
            'scores': scores}


def load_2d_data(pose: dict):
    all_points_raw = []
    all_scores = []
    for cam_name, data in pose.items():
        out = read_single_2d_data(data)
        all_points_raw.append(out['coords'])
        all_scores.append(out['scores'])

    all_points_raw = np.stack(all_points_raw, axis=1)
    all_scores = np.stack(all_scores, axis=1)

    return {'points': all_points_raw, 'scores': all_scores}


def undistort_points(all_points_raw, intrinsics: dict):
    all_points_und = np.zeros(all_points_raw.shape)

    for ix_cam, cam_name, intrinsics in enumerate(intrinsics.items()):
        calib = intrinsics[cam_name]
        points = all_points_raw[:, ix_cam].reshape(-1, 1, 2)
        points_new = cv2.undistortPoints(
            points, arr(calib['camera_mat']), arr(calib['dist_coeff']))
        all_points_und[:, ix_cam] = points_new.reshape(
            all_points_raw[:, ix_cam].shape)

    return all_points_und


def reconstruct_3d(intrinsic_dict:dict, extrinsic_3d:dict, pose_dict: dict):
    '''
    :param extrinsic_3d: list of camera matrix, aligned with camera ids
    :param pose_dict: aligned with camera ids
    :return:
    '''

    in_mat_list = []
    ex_mat_list = []
    for i, key in enumerate(intrinsic_dict.keys()):
        in_mat = np.array(intrinsic_dict[key]['camera_mat'])
        ex_mat = np.array(extrinsic_3d[str(i)])

        in_mat_list.append(in_mat)
        ex_mat_list.append(ex_mat)

    in_mat_list=np.array(in_mat_list)
    ex_mat_list=np.array(ex_mat_list).astype(float)

    out = load_2d_data(pose_dict)

    all_points = out['points']
    all_scores = out['scores']

    length = all_points.shape[0]
    shape = all_points.shape

    # preparing the containers
    all_points_3d = np.zeros((shape[0], shape[2], 3))
    all_points_3d.fill(np.nan)
    errors = np.zeros((shape[0], shape[2]))
    errors.fill(np.nan)

    scores_3d = np.zeros((shape[0], shape[2]))
    scores_3d.fill(np.nan)

    num_cams = np.zeros((shape[0], shape[2]))
    num_cams.fill(np.nan)

    all_points[all_scores < THRESHOLD] = np.nan

    # triangulate
    for i in trange(all_points.shape[0], ncols=70):
        for j in range(all_points.shape[2]):
            pts = all_points[i, :, j, :]
            good = ~np.isnan(pts[:, 0])
            if np.sum(good) >= 2:
                p3d = triangulate_simple(pts[good], ex_mat_list[good])
                all_points_3d[i, j] = p3d[:3]
                errors[i, j] = reprojection_error_und(p3d, pts[good], ex_mat_list[good], in_mat_list[good])
                num_cams[i, j] = np.sum(good)
                scores_3d[i, j] = np.min(all_scores[i, :, j][good])

    dout = pd.DataFrame()
    bp_interested = get_bp_interested(pose_dict['0'])
    for bp_num, bp in enumerate(bp_interested):
        for ax_num, axis in enumerate(['x', 'y', 'z']):
            dout[bp + '_' + axis] = all_points_3d[:, bp_num, ax_num]
        dout[bp + '_error'] = errors[:, bp_num]
        dout[bp + '_ncams'] = num_cams[:, bp_num]

    dout['fnum'] = np.arange(length)

    return dout

if __name__ =='__main__':
    import os
    import pandas as pd
    import toml

    rootpath = r'D:\Desktop\2021-03-09_h5-2053'
    processed_path=os.path.join(rootpath,'processed')
    config_path = os.path.join(rootpath,'config')

    items=os.listdir(processed_path)
    pose = [item for item in items if '.csv' in item]
    pose=sorted(pose)
    pose_dict = dict([(str(i),pd.read_csv(os.path.join(processed_path,num),header=[1,2])) for i,num in enumerate(pose)])
    length_list= [item.shape[0] for item in pose_dict.values()]
    minlen=min(length_list)-1
    for i in pose_dict.keys():
        pose_dict[i]=pose_dict[i].loc[0:minlen,:]


    items = os.listdir(config_path)
    intrinsic= [item for item in items if 'intrinsic' in item]
    intrinsic=sorted(intrinsic)
    intrinsic_dict=dict([(str(i),toml.load(os.path.join(config_path,num))) for i, num in enumerate(intrinsic)])

    extrinsic = [item for item in items if 'extrinsic' in item]
    extrinsic_dict= toml.load(os.path.join(config_path,extrinsic[0]))
    extrinsic_dict = extrinsic_dict['extrinsic']

    reconstructed_pose = reconstruct_3d(intrinsic_dict,extrinsic_dict,pose_dict)

    save_path = os.path.join(processed_path,'output_3d_data.csv')
    reconstructed_pose.to_csv(save_path)



