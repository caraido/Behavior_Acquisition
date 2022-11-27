import os
import pandas as pd

cameras_path = r'C:\\Users\\SchwartzLab\\PycharmProjects\\bahavior_rig\\DLC\\SideCamera2-Devon-2021-11-01\\labeled-data'
cameras = os.listdir(cameras_path)

for camera in cameras:
    if 'labeled' in camera:
        continue

    camera_path = os.path.join(cameras_path,camera)

    frames_and_stuff = os.listdir(camera_path)

    if 'CollectedData_Devon.h5' in frames_and_stuff:
        data = pd.read_hdf(os.path.join(camera_path, 'CollectedData_Devon.h5'))

        bad = data.index.str.contains(r'b\\img')#data[['b\\img' in s for s in data.index]]

        if bad.any():
            print(camera, 'was bad')
            data.index = data.index.str.replace(r'b\\img', r'\\img')
            data.to_hdf(os.path.join(camera_path, 'CollectedData_Devon.h5'), key='df_with_missing') # not totally sure about key
