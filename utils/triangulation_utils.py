import pandas as pd
import numpy as np
from numpy import array as arr
import cv2
from scipy import signal,optimize
from tqdm import trange
import time
from scipy.sparse import dok_matrix
from numba import jit

THRESHOLD = 0.8

def reprojection_error2(p3d, points2d, camera_mats):
    proj = np.dot(camera_mats, p3d)
    proj = proj[:, :2] / proj[:, 2, None]
    errors = np.linalg.norm(proj - points2d, axis=1)
    return np.mean(errors)


def reprojection_error(p3d, p2d,camera_mats):
    out = np.dot(p3d,camera_mats)
    out = out[:, :2] / out[:, 2, None]
    proj = out.reshape(p2d.shape)
    return p2d - proj

#@jit(nopython=True, parallel=True, forceobj=True)
def reproject_error(p3ds, p2ds, camera_mats,mean=False):
    """Given an Nx3 array of 3D points and an CxNx2 array of 2D points,
    where N is the number of points and C is the number of cameras,
    this returns an CxNx2 array of errors.
    Optionally mean=True, this averages the errors and returns array of length N of errors"""

    one_point = False
    if len(p3ds.shape) == 1 and len(p2ds.shape) == 2:
        p3ds = p3ds.reshape(1, 3)
        p2ds = p2ds.reshape(-1, 1, 2)
        one_point = True

    n_cams, n_points, _ = p2ds.shape
    assert p3ds.shape == (n_points, 3), \
        "shapes of 2D and 3D points are not consistent: " \
        "2D={}, 3D={}".format(p2ds.shape, p3ds.shape)

    errors = np.empty((n_cams, n_points, 2))

    for n_cams in range(n_cams):
        errors[n_cams] = reprojection_error(p3ds, p2ds[n_cams],camera_mats=camera_mats[n_cams])

    if mean:
        errors_norm = np.linalg.norm(errors, axis=2)
        good = ~np.isnan(errors_norm)
        errors_norm[~good] = 0
        denom = np.sum(good, axis=0).astype('float64')
        denom[denom < 1.5] = np.nan
        errors = np.sum(errors_norm, axis=0) / denom

    if one_point:
        if mean:
            errors = float(errors[0])
        else:
            errors = errors.reshape(-1, 2)

    return errors

'''
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
'''

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


def undistort_points(all_points_raw, intrinsics: dict,fisheyes:list):
    all_points_und = np.zeros(all_points_raw.shape)
    if len(intrinsics.keys()) != len(fisheyes):
        raise Exception
    else:
        for ix_cam, cam_name in enumerate(intrinsics.keys()):
            calib = intrinsics[cam_name]
            points = all_points_raw[:, ix_cam].reshape(-1, 1, 2)
            isFisheye = fisheyes[ix_cam]
            if isFisheye:
                points_new = cv2.fisheye.undistortPoints(points.astype('float64'),
                                                         arr(calib['camera_mat']).astype('float64'),
                                                         arr(calib['dist_coeff']).astype('float64'))
            else:
                points_new = cv2.undistortPoints(points.astype('float64'),
                                                 arr(calib['camera_mat']).astype('float64'),
                                                 arr(calib['dist_coeff']).astype('float64'))
            all_points_und[:, ix_cam] = points_new.reshape(
                all_points_raw[:, ix_cam].shape)

        return all_points_und

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_data(vals):
    nans, ix = nan_helper(vals)
    out = np.copy(vals)
    try:
        out[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
    except ValueError:
        out[:] = 0
    return out

def medfilt_data(values, size=15):
    padsize = size+5
    vpad = np.pad(values, (padsize, padsize), mode='reflect')
    vpadf = signal.medfilt(vpad, kernel_size=size)
    return vpadf[padsize:-padsize]

def transform_points(points, rvecs, tvecs):
    """Rotate points by given rotation vectors and translate.
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rvecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rvecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotated = cos_theta * points + \
        sin_theta * np.cross(v, points) + \
        dot * (1 - cos_theta) * v

    return rotated + tvecs

def _initialize_params_triangulation(p3ds,
                                     constraints=None,
                                     constraints_weak=None):
    if constraints_weak is None:
        constraints_weak = []
    if constraints is None:
        constraints = []
    joint_lengths = np.empty(len(constraints), dtype='float64')
    joint_lengths_weak = np.empty(len(constraints_weak), dtype='float64')

    for cix, (a, b) in enumerate(constraints):
        lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
        joint_lengths[cix] = np.median(lengths)


    for cix, (a, b) in enumerate(constraints_weak):
        lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
        joint_lengths_weak[cix] = np.median(lengths)

    all_lengths = np.hstack([joint_lengths, joint_lengths_weak])
    med = np.median(all_lengths)
    if med == 0:
        med = 1e-3

    mad = np.median(np.abs(all_lengths - med))

    joint_lengths[joint_lengths == 0] = med
    joint_lengths_weak[joint_lengths_weak == 0] = med
    joint_lengths[joint_lengths > med+mad*5] = med
    joint_lengths_weak[joint_lengths_weak > med+mad*5] = med

    return np.hstack([p3ds.ravel(), joint_lengths, joint_lengths_weak])
def _jac_sparsity_triangulation(p2ds,
                                constraints=None,
                                constraints_weak=None,
                                n_deriv_smooth=1):
    if constraints_weak is None:
        constraints_weak = []
    if constraints is None:
        constraints = []
    n_cams, n_frames, n_joints, _ = p2ds.shape
    n_constraints = len(constraints)
    n_constraints_weak = len(constraints_weak)

    p2ds_flat = p2ds.reshape((n_cams, -1, 2))

    point_indices = np.zeros(p2ds_flat.shape, dtype='int32')
    for i in range(p2ds_flat.shape[1]):
        point_indices[:, i] = i

    point_indices_3d = np.arange(n_frames*n_joints)\
                         .reshape((n_frames, n_joints))

    good = ~np.isnan(p2ds_flat)
    n_errors_reproj = np.sum(good)
    n_errors_smooth = (n_frames-n_deriv_smooth) * n_joints * 3
    n_errors_lengths = n_constraints * n_frames
    n_errors_lengths_weak = n_constraints_weak * n_frames

    n_errors = n_errors_reproj + n_errors_smooth + \
        n_errors_lengths + n_errors_lengths_weak

    n_3d = n_frames*n_joints*3
    n_params = n_3d + n_constraints + n_constraints_weak

    point_indices_good = point_indices[good]

    A_sparse = dok_matrix((n_errors, n_params), dtype='int16')

    # constraints for reprojection errors
    ix_reproj = np.arange(n_errors_reproj)
    for k in range(3):
        A_sparse[ix_reproj, point_indices_good * 3 + k] = 1

    # sparse constraints for smoothness in time
    frames = np.arange(n_frames-n_deriv_smooth)
    for j in range(n_joints):
        for n in range(n_deriv_smooth+1):
            pa = point_indices_3d[frames, j]
            pb = point_indices_3d[frames+n, j]
            for k in range(3):
                A_sparse[n_errors_reproj + pa*3 + k, pb*3 + k] = 1

    ## -- strong constraints --
    # joint lengths should change with joint lengths errors
    start = n_errors_reproj + n_errors_smooth
    frames = np.arange(n_frames)
    for cix, (a, b) in enumerate(constraints):
        A_sparse[start + cix*n_frames + frames, n_3d+cix] = 1

    # points should change accordingly to match joint lengths too
    frames = np.arange(n_frames)
    for cix, (a, b) in enumerate(constraints):
        pa = point_indices_3d[frames, a]
        pb = point_indices_3d[frames, b]
        for k in range(3):
            A_sparse[start + cix*n_frames + frames, pa*3 + k] = 1
            A_sparse[start + cix*n_frames + frames, pb*3 + k] = 1

    ## -- weak constraints --
    # joint lengths should change with joint lengths errors
    start = n_errors_reproj + n_errors_smooth + n_errors_lengths
    frames = np.arange(n_frames)
    for cix, (a, b) in enumerate(constraints_weak):
        A_sparse[start + cix*n_frames + frames, n_3d + n_constraints + cix] = 1

    # points should change accordingly to match joint lengths too
    frames = np.arange(n_frames)
    for cix, (a, b) in enumerate(constraints_weak):
        pa = point_indices_3d[frames, a]
        pb = point_indices_3d[frames, b]
        for k in range(3):
            A_sparse[start + cix*n_frames + frames, pa*3 + k] = 1
            A_sparse[start + cix*n_frames + frames, pb*3 + k] = 1

    return A_sparse

@jit(nopython=True, forceobj=True, parallel=True)
def _error_fun_triangulation(params, p2ds, cam_mats,
                             constraints=None,
                             constraints_weak=None,
                             scores=None,
                             scale_smooth=10000,
                             scale_length=1,
                             scale_length_weak=0.2,
                             reproj_error_threshold=100,
                             reproj_loss='soft_l1',
                             n_deriv_smooth=1):
    if constraints_weak is None:
        constraints_weak = []
    if constraints is None:
        constraints = []
    n_cams, n_frames, n_joints, _ = p2ds.shape

    n_3d = n_frames*n_joints*3
    n_constraints = len(constraints)
    n_constraints_weak = len(constraints_weak)

    # load params
    p3ds = params[:n_3d].reshape((n_frames, n_joints, 3))
    joint_lengths = np.array(params[n_3d:n_3d+n_constraints])
    joint_lengths_weak = np.array(params[n_3d+n_constraints:])

    # reprojection errors
    p3ds_flat = p3ds.reshape(-1, 3)
    p2ds_flat = p2ds.reshape((n_cams, -1, 2))
    errors = reproject_error(p3ds_flat, p2ds_flat,cam_mats)
    if scores is not None:
        scores_flat = scores.reshape((n_cams, -1))
        errors = errors * scores_flat[:, :, None]
    errors_reproj = errors[~np.isnan(p2ds_flat)]

    rp = reproj_error_threshold
    errors_reproj = np.abs(errors_reproj)
    if reproj_loss == 'huber':
        bad = errors_reproj > rp
        errors_reproj[bad] = rp*(2*np.sqrt(errors_reproj[bad]/rp) - 1)
    elif reproj_loss == 'linear':
        pass
    elif reproj_loss == 'soft_l1':
        errors_reproj = rp*2*(np.sqrt(1+errors_reproj/rp)-1)

    # temporal constraint
    errors_smooth = np.diff(p3ds, n=n_deriv_smooth, axis=0).ravel() * scale_smooth

    # joint length constraint
    errors_lengths = np.empty((n_constraints, n_frames), dtype='float64')
    for cix, (a, b) in enumerate(constraints):
        lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
        expected = joint_lengths[cix]
        errors_lengths[cix] = 100*(lengths - expected)/expected
    errors_lengths = errors_lengths.ravel() * scale_length

    errors_lengths_weak = np.empty((n_constraints_weak, n_frames), dtype='float64')
    for cix, (a, b) in enumerate(constraints_weak):
        lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
        expected = joint_lengths_weak[cix]
        errors_lengths_weak[cix] = 100*(lengths - expected)/expected
    errors_lengths_weak = errors_lengths_weak.ravel() * scale_length_weak

    return np.hstack([errors_reproj, errors_smooth,
                      errors_lengths, errors_lengths_weak])


def optim_points(points, p3ds,cam_mats,
                 constraints=None,
                 constraints_weak=None,
                 scale_smooth=4,
                 scale_length=2, scale_length_weak=0.5,
                 reproj_error_threshold=15, reproj_loss='soft_l1',
                 n_deriv_smooth=1, scores=None, verbose=False):
    """
    Take in an array of 2D points of shape CxNxJx2,
    an array of 3D points of shape NxJx3,
    and an array of constraints of shape Kx2, where
    C: number of camera
    N: number of frames
    J: number of joints
    K: number of constraints
    This function creates an optimized array of 3D points of shape NxJx3.
    Example constraints:
    constraints = [[0, 1], [1, 2], [2, 3]]
    (meaning that lengths of segments 0->1, 1->2, 2->3 are all constant)
    """

    if constraints_weak is None:
        constraints_weak = []
    if constraints is None:
        constraints = []
    points=points.swapaxes(0,1)
    n_cams, n_frames, n_joints, _ = points.shape
    constraints = np.array(constraints)
    constraints_weak = np.array(constraints_weak)

    p3ds_intp = np.apply_along_axis(interpolate_data, 0, p3ds)

    p3ds_med = np.apply_along_axis(medfilt_data, 0, p3ds_intp, size=7)

    default_smooth = 1.0/np.mean(np.abs(np.diff(p3ds_med, axis=0)))
    scale_smooth_full = scale_smooth * default_smooth

    t1 = time.time()

    x0 = _initialize_params_triangulation(
        p3ds_intp, constraints, constraints_weak)

    x0[~np.isfinite(x0)] = 0

    jac = _jac_sparsity_triangulation(
        points, constraints, constraints_weak, n_deriv_smooth)

    opt2 = optimize.least_squares(_error_fun_triangulation,
                                  x0=x0, jac_sparsity=jac,
                                  loss='linear',
                                  ftol=1e-3,
                                  verbose=2*verbose,
                                  args=(points,
                                        cam_mats,
                                        constraints,
                                        constraints_weak,
                                        scores,
                                        scale_smooth_full,
                                        scale_length,
                                        scale_length_weak,
                                        reproj_error_threshold,
                                        reproj_loss,
                                        n_deriv_smooth))

    p3ds_new2 = opt2.x[:p3ds.size].reshape(p3ds.shape)

    t2 = time.time()

    if verbose:
        print('optimization took {:.2f} seconds'.format(t2 - t1))

    return p3ds_new2


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

    # undistort
    all_points = out['points']
    all_scores = out['scores']
    fisheyes=[True,False,True,True]
    all_points=undistort_points(all_points,intrinsic_dict,fisheyes=fisheyes)

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
    n_cams=all_points.shape[1] # TODO:need to check
    '''
    for cam_num in range(all_points.shape[1]):
        points=all_points[:,cam_num,:,:].swapaxes(0,1)
        p3d = triangulate_ransac(points,ex_mat_list,in_mat_list[cam_num])
        #p3d = triangulate(points,ex_mat_list)
    '''

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

    # optimization
    c = np.isfinite(all_points_3d[:, :, 0])
    points_2d = all_points
    if np.sum(c) < 20:
        print("warning: not enough 3D points to run optimization")
        points_3d = all_points_3d
    else:
        points_3d = optim_points(
            points_2d, all_points_3d,
            cam_mats=in_mat_list,
            constraints=[],
            constraints_weak=[],
            # scores=scores_2d,
            scale_smooth=25,
            scale_length=10,
            scale_length_weak=2,
            n_deriv_smooth=2,
            reproj_error_threshold=3,
            verbose=True)

    points_2d_flat = points_2d.reshape(n_cams, -1, 2)
    points_3d_flat = points_3d.reshape(-1, 3)

    errors = reproject_error(
        points_3d_flat, points_2d_flat, camera_mats=in_mat_list,mean=True)
    good_points = ~np.isnan(all_points[:, :, :, 0])
    num_cams = np.sum(good_points, axis=1).astype('float')

    all_points_3d = points_3d
    all_errors = errors.reshape(shape[0], shape[1])

    all_scores[~good_points] = 2
    scores_3d = np.min(all_scores, axis=1)

    scores_3d[num_cams < 1] = np.nan
    all_errors[num_cams < 1] = np.nan

    # process the results to save
    dout = pd.DataFrame()
    bp_interested = get_bp_interested(pose_dict['0'])
    for bp_num, bp in enumerate(bp_interested):
        for ax_num, axis in enumerate(['x', 'y', 'z']):
            dout[bp + '_' + axis] = points_3d[:, bp_num, ax_num]
        dout[bp + '_error'] = all_errors[:, bp_num]
        dout[bp + '_ncams'] = num_cams[:, bp_num]

    dout['fnum'] = np.arange(length)

    return dout

if __name__ =='__main__':
    import os
    import pandas as pd
    import toml

    rootpath = r'D:\Desktop\2021-03-09_h5-2053'
    #processed_path=os.path.join(rootpath,'processed')
    processed_path=rootpath
    config_path = os.path.join(rootpath,'config')

    items=os.listdir(processed_path)
    pose = [item for item in items if '.csv' in item]
    pose=sorted(pose)
    pose_dict = dict([(str(i),pd.read_csv(os.path.join(processed_path,num),header=[1,2])) for i,num in enumerate(pose)])
    length_list= [item.shape[0] for item in pose_dict.values()]
    minlen=min(length_list)-1
    for i in pose_dict.keys():
        pose_dict[i]=pose_dict[i].loc[0:minlen,:]
        try:
            pose_dict[i] = pose_dict[i].drop(['leftarm', 'rightarm'], axis=1)
        except: pass


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



