import cv2
import numpy as np
from tqdm import trange,tqdm
import toml
from utils.calibration_utils import Calib
from multiprocess import Pool

intrinsics = [toml.load(path) for path in [
    'config/config_intrinsic_17391304.toml',
    'config/config_intrinsic_17391290.toml',
    'config/config_intrinsic_19412282.toml',
    'config/config_intrinsic_21340171.toml',
]]
for i in range(len(intrinsics)):
    intrinsics[i]['camera_mat'] = np.array(intrinsics[i]['camera_mat'])
    intrinsics[i]['dist_coeff'] = np.array(intrinsics[i]['dist_coeff'])

camera_mats = np.array([i['camera_mat'] for i in intrinsics])
dist_coeffs = [i['dist_coeff'] for i in intrinsics]
    
board = Calib('extrinsic').board
board_dict = board.dictionary
params = cv2.aruco.DetectorParameters_create()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 103
params.adaptiveThreshWinSizeStep = 4
params.adaptiveThreshConstant = 2
# params.detectInvertedMarker = True

# @jit(nopython=True)
def get_corners(frame, camera_mat, dist_coeff):
#     valid = False
    markerCorners,markerIds,rejectedCorners = cv2.aruco.detectMarkers(frame, board_dict, parameters=params)
    markerCorners,markerIds, rejectedCorners,_ = cv2.aruco.refineDetectedMarkers(
        frame, board, markerCorners, markerIds,rejectedCorners,
        camera_mat, dist_coeff,
        parameters=params)
    markerCorners = np.array(markerCorners)
    rejectedCorners = np.array(rejectedCorners)
    
    if len(markerCorners):
        #get the checkboard corners
        _, checkerCorners, checkerIds = cv2.aruco.interpolateCornersCharuco(
            markerCorners, markerIds, frame, board)
        
        if checkerCorners is None:
            checkerCorners = []
    else:
        checkerCorners = []
        checkerIds = []
    
    return markerCorners, checkerCorners, rejectedCorners, markerIds, checkerIds


def draw_corners(frame,intrinsic):
    markerCorners,checkerCorners,rejectedCorners,markerIds,checkerIds = get_corners(frame, intrinsic['camera_mat'], intrinsic['dist_coeff'])
    
    if len(checkerCorners):
        #draw the checkerboard corners
        frame = cv2.aruco.drawDetectedCornersCharuco(frame, checkerCorners, cornerColor = (0,255,0))
    
    #draw the rejected corners
    frame = cv2.aruco.drawDetectedCornersCharuco(frame, rejectedCorners, cornerColor = (255,0,0))
    
    #draw the qr code markers
    frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds, borderColor = (0,255,255))
        
    return frame, checkerCorners, checkerIds

def undistort_points(pts, camera_mat, dist_coeff):
    if len(dist_coeff) == 4:
        #case fisheye
        fn = cv2.fisheye.undistortPoints
    else:
        fn = cv2.undistortPoints
    return fn(np.ascontiguousarray(pts.reshape((-1,1,2))), camera_mat, dist_coeff)

def estimate_pose(markerCorners,ids,camera_mat,dist_coeff):
#     if not len(markerCorners):
#         return False,None,None
    
#     ret, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
# 	# 		detectedCorners, detectedIds, gray, board)
    
    undistorted = undistort_points(markerCorners, camera_mat, dist_coeff)
    camera_mat = np.eye(3)
    dist_coeffs = np.zeros((5))
    rvec = np.empty((3,1))
    tvec = np.empty((3,1))
    return cv2.aruco.estimatePoseCharucoBoard(undistorted,ids,board,cameraMatrix=camera_mat,distCoeffs=dist_coeffs,rvec=rvec,tvec=tvec,useExtrinsicGuess=False)
    

def estimate_pose_frame(first, last):#frames, camera_mats, dist_coeffs):#, rvecs, tvecs, pairs):
    n = last - first
        
    vids = [cv2.VideoCapture(i) for i in [
        'config/config_extrinsic_17391304.MOV',
        'config/config_extrinsic_17391290.MOV',
        'config/config_extrinsic_19412282.MOV',
        'config/config_extrinsic_21340171.MOV'
    ]]
    n_vids = len(vids)

    rvecs = np.empty((n,n_vids,3,1))
    tvecs = np.empty(rvecs.shape)
    pairs = np.empty(n)
    pair_order = list(range(1,n_vids))
    
    for vid in vids:
        vid.set(1,first)
    
    if first == 0:
        loop = trange
    else:
        loop = range

    for i in loop(n):
        _,corners,_,_,ids = get_corners(cv2.cvtColor(vids[0].read()[1],cv2.COLOR_BGR2GRAY), camera_mats[0], dist_coeffs[0])
        if len(corners)>3 and any(ids % 2) and not all(ids % 2):
            ret,rvecs[i,0],tvecs[i,0] = estimate_pose(corners,ids,camera_mats[0], dist_coeffs[0])
            if not ret: continue
        else: continue
        
        matched = False

        for j in pair_order:
            if matched:
                vids[j].read()
                continue
                
            _,corners,_,_,ids = get_corners(vids[j].read()[1], camera_mats[j], dist_coeffs[j])

            if len(corners)>3 and any(ids % 2) and not all(ids % 2):
                ret,rvecs[i,j],tvecs[i,j] = estimate_pose(corners,ids,camera_mats[j], dist_coeffs[j])
            else:
                continue

            if ret:
                matched = True
                pairs[i] = j
                pair_order.insert(0,pair_order.pop(pair_order.index(j)))
                
    return pairs, rvecs, tvecs


def all_extrinsics(vids):
    N_WORKERS = 7

    frame_count = int(min([c.get(cv2.CAP_PROP_FRAME_COUNT) for c in vids]))
    nvids = len(vids)
    
    ranges = np.linspace(0,int(min([c.get(cv2.CAP_PROP_FRAME_COUNT) for c in vids])),N_WORKERS + 1).astype(int)
    ranges = [(ranges[i-1], ranges[i]) for i in range(1,len(ranges))]
    print(ranges)
    
    with Pool(N_WORKERS) as p:
        out = p.starmap(estimate_pose_frame, ranges)#,vids, camera_mats, dist_coeffs),# rvecs, tvecs, pairs), 
          
    # out = estimate_pose_frame(0,frame_count)
    out = list(zip(*out))
    return np.concatenate(out[0],axis=0), np.concatenate(out[1],axis=0), np.concatenate(out[2],axis=0)

if __name__ == "__main__":
    vids = [cv2.VideoCapture(i) for i in [
        'config/config_extrinsic_17391304.MOV',
        'config/config_extrinsic_17391290.MOV',
        'config/config_extrinsic_19412282.MOV',
        'config/config_extrinsic_21340171.MOV'
    ]]
    p,r,t = all_extrinsics(vids)
    np.save('pairs2.npy',p)
    np.save('rvecs2.npy',r)
    np.save('tvecs2.npy',t)
    