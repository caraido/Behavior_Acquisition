import numpy as np


def intersect_ray_circle(head_center, arena_center,window_corner, r):
    # the ray- circle intersection problem defines a quadratic equation where the roots are the intersection points
    # ray: H+t(W-H)
    # circle: A+r(cos(t),sin(t))
    # for at^2+bt+c=0
    # a = (Wx-Hx)^2+(Wy-Hy)^2
    # b = 2[(Hx-Ax)(Wx-Hx)+(Hy-Ay)(Wy-Hy)]
    # c = (Hx-Ax)^2+(Hy-Ay)^2-r^2
    # look for b^2-4ac

    a=(window_corner[0]-head_center[:,0])**2+(window_corner[1]-head_center[:,1])**2
    b=2*np.sum((head_center-arena_center)*(window_corner-head_center),axis=1)
    c=np.sum((head_center-arena_center)**2,axis=1)-r**2
    hat = b*b-4*a*c
    visibility=hat.copy()
    visibility[hat<0]=1 # no intersection. visible
    visibility[hat>=0]=0 # intersection. not visible
    return visibility


def window_visibility(pose,windows, arena_center,r):
    left_ear = np.stack((
        pose['leftear']['x'],
        pose['leftear']['y'],
    )).transpose()  # should be t-by-2
    right_ear = np.stack((
        pose['rightear']['x'],
        pose['rightear']['y'],
    )).transpose()  # should be t-by-2

    head_center = (left_ear + right_ear) / 2
    in_windowA=np.logical_or(intersect_ray_circle(head_center,arena_center,windows[0][0],r),
                              intersect_ray_circle(head_center,arena_center,windows[0][1],r))

    in_windowB = np.logical_or(intersect_ray_circle(head_center, arena_center, windows[1][0], r),
                                intersect_ray_circle(head_center, arena_center, windows[1][1], r))

    in_windowC = np.logical_or(intersect_ray_circle(head_center, arena_center, windows[1][0], r),
                                intersect_ray_circle(head_center, arena_center, windows[1][1], r))
    in_donut=np.linalg.norm(head_center-arena_center,axis=1)-r
    in_experiment=in_donut.copy()
    in_experiment[in_donut<0]=0
    in_experiment[in_donut>0]=1

    in_windowA = np.logical_and(in_windowA,in_experiment)
    in_windowB = np.logical_and(in_windowB, in_experiment)
    in_windowC = np.logical_and(in_windowC, in_experiment)

    return in_windowA,in_windowB,in_windowC









