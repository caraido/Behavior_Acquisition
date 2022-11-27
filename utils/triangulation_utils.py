import pandas as pd
import numpy as np
from numpy import array as arr
import cv2
from scipy import signal,optimize
from tqdm import trange
import time
from scipy.sparse import dok_matrix
from numba import jit
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import itertools
from global_settings import TRI_THRESHOLD



def project(p3d,in_mat,ex_mat):
    points=p3d.reshape(-1,1,3)
    matrix = in_mat['camera_mat']
    dist = in_mat['dist_coeff']
    r_matrix = ex_mat[0:3,0:3]
    tvec = ex_mat[0:3,3]
    rvec_process =R.from_matrix(r_matrix)
    rvec = rvec_process.as_rotvec()
    if len(dist)==4:
        out,_ = cv2.fisheye.projectPoints(points,rvec,tvec,
                             matrix.astype('float64'),
                             dist.astype('float64'))
        out[out<0]=np.nan
        #out=cv2.fisheye.undistortPoints(out,matrix.astype('float64'),dist.astype('float64')) #TODO: temporarily disabled
    else:
        out,_ =cv2.projectPoints(points,rvec,tvec,
                             matrix.astype('float64'),
                             dist.astype('float64'))
        out[out<0]=np.nan
        #out = cv2.undistortPoints(out, matrix.astype('float64'),dist.astype('float64')) #TODO: temporarily disabled
    return out
'''
def reprojection_error(p3d, p2d, in_mat,ex_mat):
    proj = project(p3d,in_mat,ex_mat).reshape(p2d.shape)
    return p2d-proj
'''
def reprojection_error(p3d, p2d, in_mat,ex_mat):
    proj = project(p3d,in_mat,ex_mat).reshape(p2d.shape)
    matrix = in_mat['camera_mat']
    dist = in_mat['dist_coeff']
    if len(dist)==4:
        p2d=cv2.fisheye.distortPoints(p2d,matrix.astype('float64'),dist.astype('float64'))
    else:
        p2d = cv2.distortPoints(p2d, matrix.astype('float64'), dist.astype('float64'))
    return p2d-proj

#@jit(nopython=True, parallel=True, forceobj=True)
def reproject_error(p3ds, p2ds, in_mats,ex_mats,mean=False):
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
        errors[n_cams] = reprojection_error(p3ds, p2ds[n_cams],
                                            in_mat=in_mats[n_cams],
                                            ex_mat=ex_mats[n_cams])

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

def reprojection_error2(p3d, points2d, camera_mats):
    p3d=np.concatenate([p3d,np.array([1])])
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
        err = reprojection_error2(p3d, points[good], cam_mats[good])
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
    return p3d[0:3]

def triangulate(points, cam_mats,progress=False):
    """Given an CxNx2 array, this returns an Nx3 array of points,
    where N is the number of points and C is the number of cameras"""

    one_point = False
    if len(points.shape) == 2:
        points = points.reshape(-1, 1, 2)
        one_point = True


    n_cams, n_points, _ = points.shape

    out = np.empty((n_points, 3))
    out[:] = np.nan

    #cam_mats = np.array([cam.get_extrinsics_mat() for cam in self.cameras])

    if progress:
        iterator = trange(n_points, ncols=70)
    else:
        iterator = range(n_points)

    for ip in iterator:
        subp = points[:, ip, :]
        good = ~np.isnan(subp[:, 0])
        if np.sum(good) >= 2:
            out[ip] = triangulate_simple(subp[good], cam_mats[good])

    if one_point:
        out = out[0]

    return out

def triangulate_ransac(points, in_mats,ex_mats, min_cams=2, progress=False):
    """Given an CxNx2 array, this returns an Nx3 array of points,
    where N is the number of points and C is the number of cameras"""

    n_cams, n_points, _ = points.shape

    points_ransac = points.reshape(n_cams, n_points, 1, 2)

    return triangulate_possible(points_ransac,
                                     in_mats=in_mats,
                                     ex_mats=ex_mats,
                                     min_cams=min_cams,
                                     progress=progress)

def triangulate_possible(points, in_mats,ex_mats,
                             min_cams=2, progress=False, threshold=0.5):
        """Given an CxNxPx2 array, this returns an Nx3 array of points
        by triangulating all possible points and picking the ones with
        best reprojection error
        where:
        C: number of cameras
        N: number of points
        P: number of possible options per point
        """

        n_cams, n_points, n_possible, _ = points.shape

        cam_nums, point_nums, possible_nums = np.where(
            ~np.isnan(points[:, :, :, 0]))

        all_iters = defaultdict(dict)

        for cam_num, point_num, possible_num in zip(cam_nums, point_nums,
                                                    possible_nums):
            if cam_num not in all_iters[point_num]:
                all_iters[point_num][cam_num] = []
            all_iters[point_num][cam_num].append((cam_num, possible_num))

        for point_num in all_iters.keys():
            for cam_num in all_iters[point_num].keys():
                all_iters[point_num][cam_num].append(None)

        out = np.full((n_points, 3), np.nan, dtype='float64')
        picked_vals = np.zeros((n_cams, n_points, n_possible), dtype='bool')
        errors = np.zeros(n_points, dtype='float64')
        points_2d = np.full((n_cams, n_points, 2), np.nan, dtype='float64')

        if progress:
            iterator = trange(n_points, ncols=70)
        else:
            iterator = range(n_points)

        for point_ix in iterator:
            best_point = None
            best_error = 800

            n_cams_max = len(all_iters[point_ix])

            for picked in itertools.product(*all_iters[point_ix].values()):
                picked = [p for p in picked if p is not None]
                if len(picked) < min_cams and len(picked) != n_cams_max:
                    continue

                cnums = [p[0] for p in picked]
                xnums = [p[1] for p in picked]

                pts = points[cnums, point_ix, xnums]

                p3d = triangulate(pts, cam_mats=ex_mats[cnums])
                err = reproject_error(p3d, pts, in_mats=in_mats,ex_mats=ex_mats,mean=True)

                if err < best_error:
                    best_point = {
                        'error': err,
                        'point': p3d[:3],
                        'points': pts,
                        'picked': picked,
                        'joint_ix': point_ix
                    }
                    best_error = err
                    if best_error < threshold:
                        break

            if best_point is not None:
                out[point_ix] = best_point['point']
                picked = best_point['picked']
                cnums = [p[0] for p in picked]
                xnums = [p[1] for p in picked]
                picked_vals[cnums, point_ix, xnums] = True
                errors[point_ix] = best_point['error']
                points_2d[cnums, point_ix] = best_point['points']

        return out, picked_vals, points_2d, errors

def distort_points_cams(points, camera_mats):
    out = []
    for i in range(len(points)):
        point = np.append(points[i], 1)
        mat = camera_mats[i]
        new = mat.dot(point)[:2]
        out.append(new)
    return np.array(out)


def reprojection_error_und(p3d, points2d, camera_mats, camera_mats_dist):
    p3d=np.append(p3d,[1])
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


def load_constraints(constraints_names, bodyparts, key='constraints'):
    #constraints_names = config['triangulation'].get(key, [])
    bp_index = dict(zip(bodyparts, range(len(bodyparts))))
    constraints = []
    for a, b in constraints_names:
        assert a in bp_index, 'Bodypart {} from constraints not found in list of bodyparts'.format(a)
        assert b in bp_index, 'Bodypart {} from constraints not found in list of bodyparts'.format(b)
        con = [bp_index[a], bp_index[b]]
        constraints.append(con)
    return constraints


def read_single_2d_data(data: pd.DataFrame):
    length = len(data.index)
    index = arr(data.index)

    bp_interested = get_bp_interested(data)
    #bp_interested=['nose', 'left_ear', 'right_ear', 'tail_base']

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
                                                         arr(calib['dist_coeff']).astype('float64'),
                                                         )
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

#@jit(nopython=True, forceobj=True, parallel=True)
def _error_fun_triangulation(params, p2ds, in_mats,ex_mats,
                             constraints=None,
                             constraints_weak=None,
                             scores=None,
                             scale_smooth=100,
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
    errors = reproject_error(p3ds_flat, p2ds_flat,in_mats,ex_mats)
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


def optim_points(points, p3ds,in_mats,ex_mats,
                 constraints=None,
                 constraints_weak=None,
                 scale_smooth=4,
                 scale_length=2, scale_length_weak=0.5,
                 reproj_error_threshold=15, reproj_loss='linear',
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
                                        in_mats,
                                        ex_mats,
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
    bodypart = ['nose', 'left_ear', 'right_ear', 'tail_base']

    in_mat_list = []
    ex_mat_list = []
    for i, key in enumerate(intrinsic_dict.keys()):
        #in_mat = np.array(intrinsic_dict[key]['camera_mat'])
        cam_mat = np.array(intrinsic_dict[key]['camera_mat'])
        dist=np.array(intrinsic_dict[key]['dist_coeff'])
        in_mat = {'camera_mat':cam_mat,
                  'dist_coeff':dist}
        ex_mat = np.array(extrinsic_3d[str(i)]).astype('float64')

        in_mat_list.append(in_mat)
        ex_mat_list.append(ex_mat)

    in_mat_list=np.array(in_mat_list)
    ex_mat_list=np.array(ex_mat_list)
    # ex_mat_list[1][0:3,3]=np.array([-.9, -1.6, -.55351041])

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


    # triangulate
    for i in trange(all_points.shape[0], ncols=70):
        for j in range(all_points.shape[2]):
            pts = all_points[i, :, j, :]
            good = ~np.isnan(pts[:, 0])
            if np.sum(good) >= 2:
                p3d = triangulate_simple(pts[good], ex_mat_list[good])
                all_points_3d[i, j] = p3d[:3]
                #errors[i, j] = reprojection_error_und(p3d, pts[good], ex_mat_list[good], in_mat_list[good])
                num_cams[i, j] = np.sum(good)
                scores_3d[i, j] = np.min(all_scores[i, :, j][good])

    # get constraints
    constaints_name=[['left_ear','right_ear']]
    weak_constraints_name = [['left_ear','right_ear'],['nose','left_ear'],['nose','right_ear']]
    bodypart =['nose','left_ear','right_ear','tail_base']
    constraints = load_constraints(constaints_name,bodypart)
    weak_constraints= load_constraints(weak_constraints_name,bodypart)

    # optimization

    c = np.isfinite(all_points_3d[:, :, 0])
    points_2d = all_points
    if np.sum(c) < 20:
        print("warning: not enough 3D points to run optimization")
        points_3d = all_points_3d
    else:
        points_3d = optim_points(
            points_2d, all_points_3d,
            in_mats=in_mat_list,
            ex_mats=ex_mat_list,
            constraints=[],
            constraints_weak=weak_constraints,
            scores=all_scores,
            scale_smooth=1,
            scale_length=2,
            scale_length_weak=1,
            n_deriv_smooth=2,
            reproj_error_threshold=2,
            verbose=True)

    points_2d_flat = points_2d.reshape(n_cams, -1, 2)
    points_3d_flat = points_3d.reshape(-1, 3)

    errors = reproject_error(
        points_3d_flat, points_2d_flat, in_mats=in_mat_list,ex_mats=ex_mat_list,mean=True)
    good_points = ~np.isnan(all_points[:, :, :, 0])
    num_cams = np.sum(good_points, axis=1).astype('float')

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

# entrance for 3d smoothing
def reconstruct_3d_kalman(intrinsic_dict:dict, extrinsic_3d:dict, pose_dict:dict,all_points_3d):
    '''
    :param extrinsic_3d: list of camera matrix, aligned with camera ids
    :param pose_dict: aligned with camera ids
    :return:
    '''


    in_mat_list = []
    ex_mat_list = []
    for i, key in enumerate(intrinsic_dict.keys()):
        #in_mat = np.array(intrinsic_dict[key]['camera_mat'])
        cam_mat = np.array(intrinsic_dict[key]['camera_mat'])
        dist=np.array(intrinsic_dict[key]['dist_coeff'])
        in_mat = {'camera_mat':cam_mat,
                  'dist_coeff':dist}
        ex_mat = np.array(extrinsic_3d[str(i)]).astype('float64')

        in_mat_list.append(in_mat)
        ex_mat_list.append(ex_mat)

    in_mat_list=np.array(in_mat_list)
    ex_mat_list=np.array(ex_mat_list)
    # ex_mat_list[1][0:3,3]=np.array([-.9, -1.6, -.55351041])

    out = load_2d_data(pose_dict)

    # undistort
    all_points = out['points']
    all_scores = out['scores']
    fisheyes = [True, False, True, True]
    #all_points = undistort_points(all_points, intrinsic_dict, fisheyes=fisheyes) # TODO: temporarily disabled

    length = all_points.shape[0]
    shape = all_points.shape

    # preparing the containers
    #all_points_3d = np.zeros((shape[0], shape[2], 3))
    #all_points_3d.fill(np.nan)
    errors = np.zeros((shape[0], shape[2]))
    errors.fill(np.nan)

    scores_3d = np.zeros((shape[0], shape[2]))
    scores_3d.fill(np.nan)

    num_cams = np.zeros((shape[0], shape[2]))
    num_cams.fill(np.nan)

    all_points[all_scores < TRI_THRESHOLD] = np.nan
    n_cams = all_points.shape[1]  # TODO:need to check

    # get constraints
    constaints_name=[['left_ear','right_ear']]
    weak_constraints_name = [['left_ear','right_ear'],['nose','left_ear'],['nose','right_ear']]
    bodypart =['nose','left_ear','right_ear','tail_base']
    constraints = load_constraints(constaints_name,bodypart)
    weak_constraints= load_constraints(weak_constraints_name,bodypart)

    # optimization

    c = np.isfinite(all_points_3d[:, :, 0])
    points_2d = all_points
    if np.sum(c) < 20:
        print("warning: not enough 3D points to run optimization")
        points_3d = all_points_3d
    else:
        points_3d = optim_points(
            points_2d, all_points_3d,
            in_mats=in_mat_list,
            ex_mats=ex_mat_list,
            constraints=[],
            constraints_weak=[],
            scores=all_scores,
            scale_smooth=0,
            scale_length=2,
            scale_length_weak=2,
            n_deriv_smooth=2,
            reproj_error_threshold=5,
            verbose=True)

    points_2d_flat = points_2d.reshape(n_cams, -1, 2)
    points_3d_flat = points_3d.reshape(-1, 3)

    errors = reproject_error(
        points_3d_flat, points_2d_flat, in_mats=in_mat_list,ex_mats=ex_mat_list,mean=True)
    good_points = ~np.isnan(all_points[:, :, :, 0])
    num_cams = np.sum(good_points, axis=1).astype('float')

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

    rootpath = r'D:\Desktop\2021-04-27_Tf3-2050'
    processed_path=os.path.join(rootpath,'DLC')
    #processed_path=rootpath
    config_path = os.path.join(rootpath,'config')

    items=os.listdir(processed_path)
    pose = [item for item in items if '.csv' in item and 'camera' in item]
    pose=sorted(pose)
    pose_dict = dict([(str(i),pd.read_csv(os.path.join(processed_path,num),header=[1,2,3]).center_mouse) for i,num in enumerate(pose)])
    length_list= [item.shape[0] for item in pose_dict.values()]
    minlen=min(length_list)-1
    for i in pose_dict.keys():
        pose_dict[i]=pose_dict[i].loc[0:minlen,:]
        try:
            pose_dict[i] = pose_dict[i].drop(['leftarm', 'rightarm'], axis=1)
        except: pass

    try:
        output_3d = pd.read_csv(os.path.join(rootpath,'output_3d_data_kalma.csv'),index_col=0)
        length =len(output_3d)
        bodypart = ['nose_x','nose_y','nose_z',
                    'left_ear_x', 'left_ear_y', 'left_ear_z',
                    'right_ear_x', 'right_ear_y', 'right_ear_z',
                    'tail_base_x', 'tail_base_y', 'tail_base_z',
                    ]
        output_3d_new=output_3d[bodypart]
        output_3d_new=output_3d_new.to_numpy()
        output_3d_new=output_3d_new.reshape(-1,4,3)
    except:
        output_3d_new=None
        Exception("can't find output_3d_data_kalman.csv under the folder")

    items = os.listdir(config_path)
    intrinsic= [item for item in items if 'intrinsic' in item]
    intrinsic=sorted(intrinsic)
    intrinsic_dict=dict([(str(i),toml.load(os.path.join(config_path,num))) for i, num in enumerate(intrinsic)])

    extrinsic = [item for item in items if 'extrinsic' in item]
    extrinsic_dict= toml.load(os.path.join(config_path,extrinsic[0]))
    extrinsic_dict = extrinsic_dict['extrinsic']

    if output_3d_new is None:
        reconstructed_pose = reconstruct_3d(intrinsic_dict,extrinsic_dict,pose_dict)
    else:
        reconstructed_pose= reconstruct_3d_kalman(intrinsic_dict,extrinsic_dict,pose_dict,output_3d_new)

    save_path = os.path.join(processed_path,'output_3d_data_new.csv')
    reconstructed_pose.to_csv(save_path)



