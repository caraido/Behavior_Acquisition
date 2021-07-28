import numpy as np
import cv2
import toml
import os
import pandas as pd
from sympy import Circle,Point,sympify
from utils.head_angle_analysis import project_from_head_to_walls,is_in_window
from utils.windows_visibility import window_visibility
from utils.speed_calculation import get_speed,get_smoothed_speed

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy import io as sio
from itertools import combinations
from global_settings import FRAME_RATE

def get_board_side_length_pixel(corners):
    corners=np.array(corners)
    shape = corners.shape
    length = []
    for d in range(shape[0]):
        one =corners[d][0]
        length.append(np.sqrt((one[0][0]-one[1][0])**2 + (one[0][1]-one[1][1])**2))
        length.append(np.sqrt((one[1][0] - one[2][0]) ** 2 + (one[1][1] - one[2][1]) ** 2))
        length.append(np.sqrt((one[2][0] - one[3][0]) ** 2 + (one[2][1] - one[3][1]) ** 2))
        length.append(np.sqrt((one[3][0] - one[0][0]) ** 2 + (one[3][1] - one[0][1]) ** 2))

    average_length = np.mean(length)
    return average_length


def get_r_pixel(r, corner_pixel, board):
    resize = corner_pixel/board * r
    return resize


def find_intersect_ray_circle(board1_xy, board2_xy, board1_dist, board2_dist):
    board1_xy = Point(board1_xy[0],board1_xy[1])
    board2_xy = Point(board2_xy[0], board2_xy[1])
    circle1 = Circle(board1_xy,board1_dist)
    circle2 = Circle(board2_xy, board2_dist)
    intersect = circle1.intersect(circle2)
    intersect = np.array(intersect.args,dtype=float)
    return intersect


def find_intersect_circle_circle(circle_center1,r_1,circle_center2,r_2):
    point1 = Point(circle_center1)
    point2 = Point(circle_center2)
    circle1 = Circle(point1,sympify(str(r_1),rational=True))
    circle2 = Circle(point2,sympify(str(r_2),rational=True))
    intersect = circle2.intersect(circle1)
    if len(intersect) == 0:
        return np.nan*np.eye(2)
    else:
        intersect = np.array([intersect.args[0],intersect.args[1]],dtype=float)
        return intersect


def rank_distance(points:np.ndarray):
    distance=[]
    point=[]
    for i in range(len(points)):
        for j in range(len(points)-i-1):
            distance.append(np.sqrt(np.square(points[i][0]-points[i+j+1][0])+np.square(points[i][0]-points[i+j+1][0])))
            point.append((i,i+j+1))
    sort = np.array([a for _,a in sorted(zip(distance,point))])
    top4 = sort[0:4].reshape(-1)
    rank = np.bincount(top4)
    indices = [index for index,a in enumerate(rank) if a==2 or a==3]
    if len(indices) != 3:
        raise Exception("wrong distance")
    else:
        return points[indices]


def get_mean_circle_center(points:list):
    points = list(combinations(points,3))
    all_centers=[]
    for item in points:
        all_centers.append(get_circle_center(item[0],item[1],item[2]))
    mean_center=np.mean(all_centers,axis=0)

    return mean_center


def get_circle_center(p1, p2, p3):
    x21 = p2[0] - p1[0]
    y21 = p2[1] - p1[1]
    x32 = p3[0] - p2[0]
    y32 = p3[1] - p2[1]
    # three colinear
    if (x21 * y32 - x32 * y21 == 0):
        return None
    xy21 = p2[0] * p2[0] - p1[0] * p1[0] + p2[1] * p2[1] - p1[1] * p1[1]
    xy32 = p3[0] * p3[0] - p2[0] * p2[0] + p3[1] * p3[1] - p2[1] * p2[1]
    y0 = (x32 * xy21 - x21 * xy32) / (2 * (y21 * x32 - y32 * x21))
    x0 = (xy21 - 2 * y0 * y21) / (2.0 * x21)
    return x0, y0



def find_board_center_and_windows(corners:np.ndarray,rig:dict):
    boardA = corners[[0,3],0]
    boardB = corners[[1,4],0]
    boardC = corners[[2,5],0]
    boardA_center = np.array([np.mean(boardA[:,:,0]),np.mean(boardA[:,:,1])])
    boardB_center = np.array([np.mean(boardB[:, :, 0]), np.mean(boardB[:, :, 1])])
    boardC_center = np.array([np.mean(boardC[:, :, 0]), np.mean(boardC[:, :, 1])])
    board_length = get_board_side_length_pixel(corners)

    A2A = [get_r_pixel(rig['board_A2A'][0],board_length,rig['board_size']),get_r_pixel(rig['board_A2A'][1],board_length,rig['board_size'])]
    A2B = [get_r_pixel(rig['board_A2B'][0], board_length, rig['board_size']),get_r_pixel(rig['board_A2B'][1],board_length,rig['board_size'])]
    A2C = [get_r_pixel(rig['board_A2C'][0], board_length, rig['board_size']),get_r_pixel(rig['board_A2C'][1],board_length,rig['board_size'])]

    B2A = [get_r_pixel(rig['board_B2A'][0], board_length, rig['board_size']),get_r_pixel(rig['board_B2A'][1],board_length,rig['board_size'])]
    B2B = [get_r_pixel(rig['board_B2B'][0], board_length, rig['board_size']),get_r_pixel(rig['board_B2B'][1],board_length,rig['board_size'])]
    B2C = [get_r_pixel(rig['board_B2C'][0], board_length, rig['board_size']),get_r_pixel(rig['board_B2C'][1],board_length,rig['board_size'])]

    C2A = [get_r_pixel(rig['board_C2A'][0], board_length, rig['board_size']),get_r_pixel(rig['board_C2A'][1],board_length,rig['board_size'])]
    C2B = [get_r_pixel(rig['board_C2B'][0], board_length, rig['board_size']),get_r_pixel(rig['board_C2B'][1],board_length,rig['board_size'])]
    C2C = [get_r_pixel(rig['board_C2C'][0], board_length, rig['board_size']),get_r_pixel(rig['board_C2C'][1],board_length,rig['board_size'])]


    winA1 = np.array([find_intersect_circle_circle(boardA_center,A2A[0],boardB_center,B2A[0]), \
                      find_intersect_circle_circle(boardA_center, A2A[0], boardC_center, C2A[0]), \
                      find_intersect_circle_circle(boardC_center, C2A[0], boardB_center, B2A[0])]).reshape([-1,2])
    winA2 = np.array([find_intersect_circle_circle(boardA_center,A2A[1],boardB_center,B2A[1]), \
                      find_intersect_circle_circle(boardA_center, A2A[1], boardC_center, C2A[1]), \
                      find_intersect_circle_circle(boardC_center, C2A[1], boardB_center, B2A[1])]).reshape([-1,2])

    winB1 = np.array([find_intersect_circle_circle(boardA_center, A2B[0], boardB_center, B2B[0]), \
                      find_intersect_circle_circle(boardA_center, A2B[0], boardC_center, C2B[0]), \
                      find_intersect_circle_circle(boardC_center, C2B[0], boardB_center, B2B[0])]).reshape([-1,2])
    winB2 = np.array([find_intersect_circle_circle(boardA_center, A2B[1], boardB_center, B2B[1]), \
                      find_intersect_circle_circle(boardA_center, A2B[1], boardC_center, C2B[1]), \
                      find_intersect_circle_circle(boardC_center, C2B[1], boardB_center, B2B[1])]).reshape([-1,2])

    winC1 = np.array([find_intersect_circle_circle(boardA_center, A2C[0], boardB_center, B2C[0]), \
                      find_intersect_circle_circle(boardA_center, A2C[0], boardC_center, C2C[0]), \
                      find_intersect_circle_circle(boardC_center, C2C[0], boardB_center, B2C[0])]).reshape([-1,2])
    winC2 = np.array([find_intersect_circle_circle(boardA_center, A2C[1], boardB_center, B2C[1]), \
                      find_intersect_circle_circle(boardA_center, A2C[1], boardC_center, C2C[1]), \
                      find_intersect_circle_circle(boardC_center, C2C[1], boardB_center, B2C[1])]).reshape([-1,2])

    A1 = np.mean(rank_distance(winA1),axis=0)
    A2 = np.mean(rank_distance(winA2),axis=0)

    B1 = np.mean(rank_distance(winB1),axis=0)
    B2 = np.mean(rank_distance(winB2),axis=0)

    C1 = np.mean(rank_distance(winC1),axis=0)
    C2 = np.mean(rank_distance(winC2),axis=0)

    circle_center = get_mean_circle_center([A1,A2,B1,B2,C1,C2])
    circle_center=circle_center.astype(np.int)
    A1=A1.astype(np.int)
    A2=A2.astype(np.int)
    B1=B1.astype(np.int)
    B2=B2.astype(np.int)
    C1=C1.astype(np.int)
    C2=C2.astype(np.int)

    results = {'recorded_center': circle_center,
                 'A1':A1,
                 'A2': A2,
                 'B1': B1,
                 'B2': B2,
                 'C1': C1,
                 'C2': C2,
                    }
    return results


def _triangle_area(point1,point2,point3):
    a=np.linalg.norm(point1-point2,axis=0)
    b=np.linalg.norm(point3-point2,axis=0)
    c=np.linalg.norm(point1-point3,axis=0)
    s=(a + b + c) / 2
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    return area

def triangle_area(pose):
    snout_x = pose['snout']['x'].to_numpy()
    snout_y = pose['snout']['y'].to_numpy()

    leftear_x = pose['leftear']['x'].to_numpy()
    leftear_y = pose['leftear']['y'].to_numpy()

    rightear_x = pose['rightear']['x'].to_numpy()
    rightear_y = pose['rightear']['y'].to_numpy()

    tailbase_x = pose['rightear']['x'].to_numpy()
    tailbase_y = pose['rightear']['y'].to_numpy()

    snout=np.array([snout_x,snout_y])
    leftear = np.array([leftear_x, leftear_y])
    rightear = np.array([rightear_x, rightear_y])
    tailbase = np.array([tailbase_x, tailbase_y])

    head_triangle_area = _triangle_area(snout,leftear,rightear)
    body_triangle_area = _triangle_area(tailbase,leftear,rightear)

    return head_triangle_area,body_triangle_area


def body_center(pose):
    leftear_x = pose['leftear']['x'].to_numpy()
    leftear_y = pose['leftear']['y'].to_numpy()

    rightear_x = pose['rightear']['x'].to_numpy()
    rightear_y = pose['rightear']['y'].to_numpy()

    center = np.array([(leftear_x+rightear_x)/2,(leftear_y+rightear_y)/2])
    return center

def window_crossings(arc_body, body_long,smoothed_speed):
    arc_body=arc_body[:,0]
    # body_long is a boolean array
    info={'window':[],
          'type':[],
          'in_frame':[],
          'out_frame':[],
          'frame_spent':[],
          'average_speed':[]
    }
    window_names=['A','B','C']
    for windex,in_window in enumerate(body_long):

        enter_side=None
        in_frame=0
        exit_side=None
        out_frame=0

        for i in range(len(in_window)-1):
            if not np.isnan(arc_body[i]):
                # frame i=0 and frame i+1=1: entering window
                if in_window[i]==0 and in_window[i+1]==1:
                    enter_side = arc_body[i+1]-arc_body[i]
                    in_frame=i+1
                # frame i=1 and frame i+1=0: exiting window
                if in_window[i]==1 and in_window[i+1]==0:
                    exit_side = arc_body[i+1]-arc_body[i]
                    out_frame=i+1

                # if the mouse exits
                if enter_side is not None and exit_side is not None and out_frame>in_frame:
                    in_window_speed=np.mean(smoothed_speed[in_frame:out_frame])
                    info['average_speed'].append(in_window_speed)
                    info['window'].append(window_names[windex])
                    info['in_frame'].append(in_frame)
                    info['out_frame'].append(out_frame)
                    info['frame_spent'].append(out_frame-in_frame)
                    if enter_side*exit_side>0: # this means straight through
                        info['type'].append('through')
                    else:
                        info['type'].append('back')

                    enter_side = None
                    in_frame = 0
                    exit_side = None
                    out_frame = 0
    return info


class Config:
    def __init__(self,path:str,save_center=False):
        self.config_folder_path=path
        self.img=None
        self.config_rig=None
        self.config_top_cam=None
        self._load_config()
        if self.config_top_cam:
            self.circle_center=np.array(self.config_top_cam['recorded_center'])
            self.window_A = np.array([self.config_top_cam['A1'],self.config_top_cam['A2']])
            self.window_B = np.array([self.config_top_cam['B1'],self.config_top_cam['B2']])
            self.window_C = np.array([self.config_top_cam['C1'],self.config_top_cam['C2']])

        self.save_center=save_center

    def set_img(self,path:str):
        self.img = cv2.imread(path)
        if self.img is None:
            cap=cv2.VideoCapture(path)
            ret =True
            while ret:
                ret,frame = cap.read()
                if ret:
                    self.img=frame
                    ret=False
            cap.release()

    def _load_config(self):
        last = os.path.split(self.config_folder_path)[-1]
        if '.' not in last:
            dir = os.listdir(self.config_folder_path)
            for item in dir:
                if 'behavior_rig' in item:
                    path = os.path.join(self.config_folder_path, item)
                    self.config_rig=toml.load(path)
                if 'alignment' in item:
                    path = os.path.join(self.config_folder_path,item)
                    self.config_top_cam = toml.load(path)
        elif '.toml' in last:
            if 'behavior_rig' in last:
                self.config_rig = toml.load(self.config_folder_path)
            if 'alignment' in last:
                self.config_top_cam = toml.load(self.config_folder_path)


        if self.config_top_cam is None or self.config_rig is None:
            print("one or both configuration file not found")

    def _save_center(self):
        if self.save_center:
            if os.path.exists(self.config_folder_path+'\\config_behavior_rig.toml'):
                with open(self.config_folder_path+'\\config_behavior_rig.toml', 'a') as f:
                    new_toml_string = toml.dump({'recorded_center':self.circle_center}, f)
                    print('saved')
                    print(new_toml_string)
            else:
                with open(self.config_folder_path+'\\config_behavior_rig.toml', 'w') as f:
                    new_toml_string = toml.dump({'recorded_center': self.circle_center}, f)
                    print('saved')
                    print(new_toml_string)

        else:
            pass

    def on_EVENT_LBUTTONDOWN(self, event, x, y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            self.circle_center=[x,y]
            self._save_center()
            print(xy)
            cv2.circle(self.img, (x, y), 1, (255, 0, 0), thickness = -1)
            cv2.putText(self.img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0,0,0), thickness = 1)
            cv2.imshow("image", self.img)

    def get_loc_pixel(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.on_EVENT_LBUTTONDOWN)
        cv2.imshow("image", self.img)
        while True:
            try:
                cv2.waitKey(100)
            except Exception:
                cv2.destroyWindow("image")
                break


def accumulative_window_preference(long):
    length=long.shape[1]
    accum=[]

    for i in range(length-1):
        seg=long[:,0:i+1]
        accum.append(np.sum(seg,axis=1))

    accum=np.array(accum)
    return accum


class Gaze_angle:
    def __init__(self, config_folder_path, gazePoint=0.5725, main_config_path=None):
        if main_config_path:
            main_config = Config(main_config_path)
            local_config = toml.load(config_folder_path)
        else:
            main_config = Config(config_folder_path)
            local_config = main_config.config_top_cam

        self.config_folder_path= config_folder_path
        self.corners = main_config.config_top_cam['undistorted_corners']
        self.inner_r = main_config.config_rig['inner_r']
        self.outer_r = main_config.config_rig['outer_r_']
        self.board_size = main_config.config_rig['board_size']

        self.windowA = np.array([local_config['A1'],local_config['A2']])
        self.windowB = np.array([local_config['B1'], local_config['B2']])
        self.windowC = np.array([local_config['C1'], local_config['C2']])
        self.circle_center = np.array(local_config['recorded_center'])
        self.windows = self._get_window_angle()
        self.pixel2inch=float(local_config['pixel2inch']) # for bottom circle
        self.inch2pixel=1/self.pixel2inch

        # recalculate the pixel length of circle radius
        corners_pixel = get_board_side_length_pixel(self.corners)
        self.inner_r_pixel = get_r_pixel(self.inner_r, corners_pixel, self.board_size)
        self.outer_r_pixel = get_r_pixel(self.outer_r, corners_pixel, self.board_size)

        self.inner=None
        self.outer=None
        self.title_name=None
        self.gazePoint=gazePoint

    def __call__(self,root_path, cutoff=0.95, save=True):
        # get the pose estimation path
        # coord_path = 'C:\\Users\\SchwartzLab\\Downloads\\Acclimation_videos_1\\'
        self.title_name = os.path.split(root_path)[-1]

        all_file = os.listdir(root_path)
        name = [a for a in all_file if 'second' in a and 'csv' in a]
        coord = os.path.join(root_path,name[0])
        pose = pd.read_csv(coord, header=[1,2])

        # cutoff (doesn't know if it works
        pose.loc[pose.leftear.likelihood < cutoff, [('leftear', 'x'), ('leftear', 'y')]] = np.nan
        pose.loc[pose.rightear.likelihood < cutoff, [('rightear', 'x'), ('rightear', 'y')]] = np.nan
        pose.loc[pose.snout.likelihood < cutoff, [('snout', 'x'), ('snout', 'y')]] = np.nan
        pose.loc[pose.tailbase.likelihood < cutoff, [('tailbase', 'x'), ('tailbase', 'y')]] = np.nan
        empty = [pose['leftear']['x'].sum(),pose['leftear']['y'].sum(),
                pose['rightear']['x'].sum(),pose['rightear']['y'].sum(),
                pose['snout']['x'].sum(),pose['snout']['y'].sum(),
                pose['tailbase']['x'].sum(),pose['tailbase']['y'].sum()]
        if not any(np.array(empty)==0):
            # gaze
            inner_left,outer_left, inner_right, outer_right = project_from_head_to_walls(pose,
                                                                                          self.inner_r_pixel,
                                                                                          self.outer_r_pixel,
                                                                                          self.circle_center[np.newaxis, :],
                                                                                          gazePoint=self.gazePoint)

            # get window visibility
            window_pixels=np.array([self.windowA,self.windowB,self.windowC])
            winA_vis,winB_vis,winC_vis=window_visibility(pose,window_pixels,self.circle_center,self.inner_r_pixel)

            # get smoothed speed over time. unit: inch/second
            speed=get_speed(pose)
            smoothed_speed=get_smoothed_speed(speed,param=31,smoothing_method='median_filter')*self.pixel2inch*FRAME_RATE

            # triangle area
            head_triangle_area,body_triangle_area = triangle_area(pose) # in squared pixel
            head_triangle_area=head_triangle_area*self.pixel2inch**2 # in squared inches
            body_triangle_area = body_triangle_area * self.pixel2inch ** 2 # in squared inches

            # body center radian
            body = body_center(pose)
            recenter_body = body-self.circle_center[:,np.newaxis]
            arc_body = np.arctan2(recenter_body[1],recenter_body[0])[:,np.newaxis]

            # check if the body center is in experiment area
            distance = np.linalg.norm(recenter_body,axis=0)
            for i,dis in enumerate(distance):
                if dis<self.inner_r_pixel or dis>self.outer_r_pixel:
                    inner_left[i]=np.nan
                    outer_left[i]=np.nan
                    inner_right[i]=np.nan
                    outer_right[i]=np.nan
                    arc_body[i]=np.nan

            # in window
            A_right = np.sum(is_in_window(outer_right,self.windowA,self.circle_center))
            B_right = np.sum(is_in_window(outer_right, self.windowB, self.circle_center))
            C_right = np.sum(is_in_window(outer_right, self.windowC, self.circle_center))

            A_right_long=is_in_window(outer_right,self.windowA,self.circle_center)
            B_right_long = is_in_window(outer_right, self.windowB, self.circle_center)
            C_right_long = is_in_window(outer_right, self.windowC, self.circle_center)
            right_long=np.stack([A_right_long, B_right_long, C_right_long])

            A_left = np.sum(is_in_window(outer_left, self.windowA, self.circle_center))
            B_left = np.sum(is_in_window(outer_left, self.windowB, self.circle_center))
            C_left = np.sum(is_in_window(outer_left, self.windowC, self.circle_center))

            A_left_long = is_in_window(outer_left, self.windowA, self.circle_center)
            B_left_long = is_in_window(outer_left, self.windowB, self.circle_center)
            C_left_long = is_in_window(outer_left, self.windowC, self.circle_center)
            left_long=np.stack([A_left_long, B_left_long, C_left_long])

            # check the amount of frame when the body is in window
            A_body = np.sum(is_in_window(arc_body, self.windowA, self.circle_center))
            B_body = np.sum(is_in_window(arc_body, self.windowB, self.circle_center))
            C_body = np.sum(is_in_window(arc_body, self.windowC, self.circle_center))

            # check if body is in window.
            A_body_long = is_in_window(arc_body, self.windowA, self.circle_center)
            B_body_long = is_in_window(arc_body, self.windowB, self.circle_center)
            C_body_long = is_in_window(arc_body, self.windowC, self.circle_center)
            body_long = np.stack([A_body_long, B_body_long, C_body_long])

            A_weight_right = A_right /  (A_right + B_right + C_right)
            B_weight_right = B_right / (A_right + B_right + C_right)
            C_weight_right = C_right / (A_right + B_right + C_right)

            A_weight_left = A_left / (A_left + B_left + C_left)
            B_weight_left = B_left / (A_left + B_left + C_left)
            C_weight_left = C_left / (A_left + B_left + C_left)

            A_weight_body = A_body / (A_body + B_body + C_body)
            B_weight_body = B_body / (A_body + B_body + C_body)
            C_weight_body = C_body / (A_body + B_body + C_body)

            window_A_center = np.array(self.windowA) - np.array(self.circle_center)
            window_B_center = np.array(self.windowB) - np.array(self.circle_center)
            window_C_center = np.array(self.windowC) - np.array(self.circle_center)

            window_A_angle = np.arctan2(window_A_center[:, 1], window_A_center[:, 0])
            window_B_angle = np.arctan2(window_B_center[:, 1], window_B_center[:, 0])
            window_C_angle = np.arctan2(window_C_center[:, 1], window_C_center[:, 0])

            windows_arc= np.abs(window_A_angle[0]-window_A_angle[1])+\
                         np.abs(window_B_angle[0]-window_B_angle[1])+\
                         np.abs(window_C_angle[0]-window_C_angle[1])

            winPreference_right = ((A_right + B_right + C_right) / np.sum(pd.notna(outer_right))) * 2*np.pi/windows_arc
            winPreference_left = ((A_left + B_left + C_left) / np.sum(pd.notna(outer_left))) * 2*np.pi/windows_arc
            winPreference_body = ((A_body + B_body + C_body) / np.sum(pd.notna(arc_body))) * 2*np.pi/windows_arc

            table = np.array([A_weight_right,B_weight_right,C_weight_right,winPreference_right,
                              A_weight_left,B_weight_left,C_weight_left,winPreference_left,
                              A_weight_body,B_weight_body,C_weight_body,winPreference_body
                              ]).reshape([3,4])
            column = ['windowA','windowB','windowC','window preference']
            index = ['right','left','body']
            stats=pd.DataFrame(table,columns=column,index=index)
            stats=stats.to_dict()

            window_crossings_data=window_crossings(arc_body,body_long,smoothed_speed)

            left_accum=accumulative_window_preference(left_long)
            right_accum=accumulative_window_preference(right_long)
            body_accum=accumulative_window_preference(body_long)

            # radian of nose circle center
            nose_circle_center_arc=np.arctan2(pose['snout']['y']-self.circle_center[1],pose['snout']['x']-self.circle_center[0])
            nose_distance = self.outer_r - np.sqrt((pose['snout']['x'] - self.circle_center[0]) ** 2 + (
                    pose['snout']['y'] - self.circle_center[1]) ** 2) * self.outer_r/self.outer_r_pixel
            #minimum_distance=min(nose_distance[nose_distance>0])
            #nose_distance=np.array(nose_distance)-minimum_distance # assume that nose always poke the outer wall

            # for windows A
            nose_distance_A=nose_distance.copy()
            for i in range(len(nose_circle_center_arc)):
                if not np.min(self.windows[0][0]) < nose_circle_center_arc[i] < np.max(self.windows[0][0]):
                    nose_distance_A[i]=np.nan

            # for windows B
            nose_distance_B = nose_distance.copy()
            for i in range(len(nose_circle_center_arc)):
                if not np.min(self.windows[1][0]) < nose_circle_center_arc[i] < np.max(self.windows[1][0]):
                    nose_distance_B[i] = np.nan

            # for windows C
            nose_distance_C = nose_distance.copy()
            for i in range(len(nose_circle_center_arc)):
                if not np.min(self.windows[2][0]) < nose_circle_center_arc[i] < np.max(self.windows[2][0]):
                    nose_distance_C[i] = np.nan

            # find minimum
            nose_distance_A_min=min(nose_distance_A[nose_distance_A>0])
            nose_distance_B_min = min(nose_distance_B[nose_distance_B > 0])
            nose_distance_C_min = min(nose_distance_C[nose_distance_C > 0])
            nose_distance_min=min([nose_distance_A_min,nose_distance_B_min,nose_distance_C_min])

            nose_distance={'window_A':np.array(nose_distance_A-nose_distance_min),
                           'window_B':np.array(nose_distance_B-nose_distance_min),
                           'window_C':np.array(nose_distance_C-nose_distance_min)}


            result = {'inner_left': np.transpose(inner_left)[0],
                      'inner_right': np.transpose(inner_right)[0],
                      'outer_left': np.transpose(outer_left)[0],
                      'outer_right': np.transpose(outer_right)[0],
                      'left_accum':left_accum,
                      'right_accum': right_accum,
                      'body_accum':body_accum,
                      'head_triangle_area':head_triangle_area,
                      'body_triangle_area':body_triangle_area,
                      'body_position':np.transpose(arc_body)[0],
                      'win_visibility':np.stack([winA_vis,winB_vis,winC_vis]),
                      'speed':smoothed_speed,
                      'stats':stats,
                      'window_arc':self.windows,
                      'nose_window_distance':nose_distance,
                      'window_crossings':window_crossings_data
                      }

            # save file
            if save:
                self.save(root_path, result)
                print('saved gaze')

            return result
        else:
            return None

    def double(self,side):
        side=side[pd.notna(side)]
        side_right = side[side > 0]
        side_left = side[side <= 0]
        side = np.append(side, side_right - 2 * np.pi)
        side = np.append(side, side_left + 2 * np.pi)
        return side

    def _get_window_angle(self):
        window_A_center = np.array(self.windowA) - np.array(self.circle_center)
        window_B_center = np.array(self.windowB) - np.array(self.circle_center)
        window_C_center = np.array(self.windowC) - np.array(self.circle_center)

        window_A_angle = np.sort(np.arctan2(window_A_center[:, 1], window_A_center[:, 0]))
        window_B_angle = np.sort(np.arctan2(window_B_center[:, 1], window_B_center[:, 0]))
        window_C_angle = np.sort(np.arctan2(window_C_center[:, 1], window_C_center[:, 0]))

        window_A_angle_extra = np.array([i - 2 * np.pi if i > 0 else i + 2 * np.pi for i in window_A_angle])
        window_B_angle_extra = np.array([i - 2 * np.pi if i > 0 else i + 2 * np.pi for i in window_B_angle])
        window_C_angle_extra = np.array([i - 2 * np.pi if i > 0 else i + 2 * np.pi for i in window_C_angle])

        windowA = [window_A_angle,window_A_angle_extra]
        windowB = [window_B_angle, window_B_angle_extra]
        windowC = [window_C_angle, window_C_angle_extra]

        combine = [windowA, windowB, windowC]

        return combine

    def save(self,path, data):
        path =os.path.join(path,'gaze')
        if not os.path.exists(path):
            os.mkdir(path)
        if isinstance(data,pd.DataFrame):
            data.to_csv(os.path.join(path,'gaze_angle_%d.csv'%int(self.gazePoint*180/np.pi)))
        else:
            filename = os.path.join(path,'gaze_angle_%d.mat'%int(self.gazePoint*180/np.pi))
            sio.savemat(filename,data)

    def plot(self,things,savepath=None,show=False):
        if things is not None:
            self.plot_gaze(things,savepath)
            plt.clf()

            # plot the accumulative gaze
            right_accum=None
            left_accum=None
            try:
                right_accum=things['right_accum']
                left_accum = things['left_accum']
            except:
                raise Warning('cannot find accumulative gaze data')
            if right_accum is not None and left_accum is not None:
                self.plot_accum(right_accum,'right',savepath)
                self.plot_accum(left_accum,'left',savepath)
            plt.clf()

            speed=None
            try:
                speed=things['speed']
            except:
                raise Warning('cannot find speed data')
            if speed is not None:
                self.plot_speed_over_time(speed, savepath)
            plt.clf()
            plt.close()

    def plot_gaze(self,things,savepath):
        keys = ['inner_left', 'inner_right', 'outer_left', 'outer_right', 'body_position']
        row = 4
        col = 2
        # plot the gaze
        #plt.ioff()
        plt.figure(figsize=(14, 16))
        for i in range(len(keys)):
            plt.subplot(row, col, i + 1)

            plt.axvspan(xmin=self.windows[0][0][0], xmax=self.windows[0][0][1], facecolor='orange', alpha=0.3)
            plt.axvspan(xmin=self.windows[1][0][0], xmax=self.windows[1][0][1], facecolor='red', alpha=0.3)
            plt.axvspan(xmin=self.windows[2][0][0], xmax=self.windows[2][0][1], facecolor='magenta', alpha=0.3)
            plt.axvspan(xmin=self.windows[0][1][0], xmax=self.windows[0][1][1], facecolor='orange', alpha=0.3)
            plt.axvspan(xmin=self.windows[1][1][0], xmax=self.windows[1][1][1], facecolor='red', alpha=0.3)
            plt.axvspan(xmin=self.windows[2][1][0], xmax=self.windows[2][1][1], facecolor='magenta', alpha=0.3)
            handles = [Rectangle((0, 0), 1, 1, color=c, ec="k", alpha=0.3) for c in ['orange', 'red', 'magenta']]
            labels = ["window A", "window B", "window C"]
            plt.legend(handles, labels)
            plt.title("Gaze point density of %s " % list(keys)[i])
            plt.hist(x=self.double(np.array(things[list(keys)[i]])), bins=60, density=False)
        plt.suptitle(self.title_name + ' gaze angle %d' % float(self.gazePoint * 180 / np.pi), fontsize=30)

        plt.subplot(4, 1, 4)
        plt.axis('off')
        stats = things['stats']
        stats = pd.DataFrame(stats)

        cell_text = []
        for row in range(len(stats)):
            cell_text.append(['%1.2f' % x for x in stats.iloc[row]])

        table = plt.table(cellText=cell_text, colLabels=stats.columns, rowLabels=stats.index, loc='center',
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(20)
        table.scale(1, 3)
        if savepath:
            plt.savefig(os.path.join(savepath, 'gaze', 'gaze_angle_%d.jpg' % float(self.gazePoint * 180 / np.pi)))
        plt.close()

    def plot_accum(self,accum,side,savepath):
        #plt.ioff()
        plt.figure()
        length=accum.shape[0]
        things=os.listdir(savepath)
        items=[i for i in things if '.MOV' in i]

        cap=cv2.VideoCapture(os.path.join(savepath,items[0]))
        fps=cap.get(cv2.CAP_PROP_FPS)
        x=np.linspace(0,length/fps/60,length) # get the x ticks in minutes\
        plt.plot(x,accum)
        plt.xlabel('time(min)')
        plt.ylabel('gaze on windwo count')
        plt.legend(['window A','window B','window C'])
        plt.title('Accumulative gaze preference for %s eye of gaze angle %d'%(side,float(self.gazePoint*180/np.pi)))
        plt.savefig(os.path.join(savepath,'gaze','accumulative_gaze_pref_%s_angle_%d'%(side,float(self.gazePoint*180/np.pi))))
        plt.close()

    def plot_speed_over_time(self,speed,savepath):
        #plt.ioff()
        plt.figure()
        length=speed.shape[0]
        things = os.listdir(savepath)
        items = [i for i in things if '.MOV' in i]

        cap = cv2.VideoCapture(os.path.join(savepath, items[0]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        x = np.linspace(0, length / fps / 60, length)  # get the x ticks in minutes\
        average_speed=np.nanmean(speed)
        plt.plot(x,speed)
        plt.xlabel('time(min)')
        plt.ylabel('speed (pixel/frame)')
        plt.ylim([0,40])
        plt.axhline(y=average_speed,color='red')
        plt.legend(['speed','average speed'])
        plt.title('Speed')
        plt.savefig(os.path.join(savepath, 'gaze','speed.jpg'))
        plt.close()


if __name__ == '__main__':
    #import matlab.engine

    #config_folder_path = r'C:\\Users\\SchwartzLab\\PycharmProjects\\bahavior_rig\\config'
    #img_path = r'C:\\Users\\SchwartzLab\\PycharmProjects\\bahavior_rig\\test.png'
    '''
    path = r'D:\Desktop\2144\2021-06-17_habituation_dominant'
    ob = Config(path, save_center=False)
    video_path = path + r'\camera_17391304.MOV'
    ob.set_img(video_path)
    ob.get_loc_pixel()


    '''
    dlc_path = r'D:\Desktop\1085\2021-07-06_mother_(A)novel_object_(B)pup_(C)pups\DLC'
    root_path = r'D:\Desktop\1085\2021-07-06_mother_(A)novel_object_(B)pup_(C)pups\config'
    gaze=Gaze_angle(root_path)
    result=gaze(dlc_path)


