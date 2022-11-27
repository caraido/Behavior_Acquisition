import cv2
import PySpin
import numpy as np
import ffmpeg
from utils.calibration_utils import Calib
import pandas as pd
from dlclive import DLCLive, Processor
from AcquisitionObject import AcquisitionObject
from utils.image_draw_utils import draw_dots
import os
import time
from global_settings import FRAME_TIMEOUT,FRAME_BUFFER,DLC_RESIZE,DLC_UPDATE_EACH,TOP_CAM,TEMP_PATH,N_BUFFER


class Camera(AcquisitionObject):

  def __init__(self, parent, camlist, index, frame_rate, address):

    self._spincam = camlist.GetByIndex(index)
    self._spincam.Init()

    # hardware triggering
    self._spincam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    # self._spincam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_SingleFrame)
    # trigger has to be off to change source
    self._spincam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    self._spincam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
    self._spincam.TriggerMode.SetValue(PySpin.TriggerMode_On)

    self.device_serial_number, self.height, self.width = self.get_camera_properties()

    # set the buffer
    self.set_buffer(nbuffer_frame=N_BUFFER)

    AcquisitionObject.__init__(
        self, parent, frame_rate, (self.width, self.height), address)
    self.is_top = True if self.device_serial_number == TOP_CAM else False

    self.save_count=0
    self.capture_count=0
    self.diplay_count = 0

  def set_buffer(self,nbuffer_frame=10): #default is 10
    ## use this to handle buffer. Example in BufferHandling.py in PySpin api
    # Retrieve Buffer Handling Mode Information
    self.nodemap_tlstream = self._spincam.GetTLStreamNodeMap()
    #self.stream_buffer_count = PySpin.CIntegerPtr(nodemap_tlstream.GetNode('StreamTotalBufferCount'))
    handling_mode = PySpin.CEnumerationPtr(self.nodemap_tlstream.GetNode('StreamBufferHandlingMode'))
    handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())

    # Set stream buffer Count Mode to manual
    stream_buffer_count_mode = PySpin.CEnumerationPtr(self.nodemap_tlstream.GetNode('StreamBufferCountMode'))
    stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
    stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
    print('Stream Buffer Count Mode set to manual...')

    # Retrieve and modify Stream Buffer Count
    buffer_count = PySpin.CIntegerPtr(self.nodemap_tlstream.GetNode('StreamBufferCountManual'))

    # Display Buffer Info
    print('Default Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
    print('Default Buffer Count: %d' % buffer_count.GetValue())
    print('Maximum Buffer Count: %d' % buffer_count.GetMax())

    buffer_count.SetValue(nbuffer_frame)

    print('Buffer count now set to: %d' % buffer_count.GetValue())

  def prepare_run(self):  # TODO: prepare_run?
    self._spincam.BeginAcquisition()

  def end_run(self):
    self._spincam.EndAcquisition()

  def end_display(self):
    self.print('Ending display for camera')

  def prepare_processing(self, options):
    process = {}

    if options['mode'] == 'DLC':
      # process['modelpath'] = options
      process['mode'] = 'DLC'
      process['processor'] = Processor()
      process['DLCLive'] = DLCLive(
          model_path=options['modelpath'],
          processor=process['processor'],
          display=False,
          resize=DLC_RESIZE,
          dynamic=(True,0.7,40))
      process['frame0'] = True
      process['frame_num']=0
      return process
    else:  # mode should be 'intrinsic' or 'extrinsic'
      process['mode'] = options['mode']

      # could move this to init if desired
      process['calibrator'] = Calib(options['mode'])
      process['calibrator'].load_in_config(self.device_serial_number)
      # TODO: is there a better to handle recording during calibration?

      return process
      # process['calibrator'].root_config_path= self.file # does this return the file path?

      # process['calibrator'].reset()
      # if options['mode'] == 'extrinsic':
      # process['calibrator'].load_ex_config(self.device_serial_number)

  def end_processing(self, process):
    if process['mode'] == 'DLC':
      process['DLCLive'].close()
      process['frame0'] = False
      status = 'DLC Live turned off'
    else:
      status = process['calibrator'].save_temp_config(
          self.device_serial_number, self.width, self.height)
      self.print(status)
      del process['calibrator']  # could move this to close if desired
    # TODO:status should be put on the screen!
    return status

  def do_process(self, data, data_count, process):
    if process['mode'] == 'DLC':
      process['frame_num'] = process['frame_num'] + 1
      if process['frame0']:
        process['DLCLive'].init_inference(frame=data)
        process['frame0'] = False
        pose = process['DLCLive'].get_pose(data)
        return pose, process
      else:
        return process['DLCLive'].get_pose(data), None
    elif process['mode'] == 'intrinsic':
      result = process['calibrator'].in_calibrate(
          data, data_count, self.device_serial_number)
      return result, None

    elif process['mode'] == 'alignment':
      result = process['calibrator'].al_calibrate(data, data_count)
      return result, None

    elif process['mode'] == 'extrinsic':
      result = process['calibrator'].ex_calibrate(data, data_count)
      return result, None

  def run(self):
    if self._has_runner:
      return  # only 1 runner at a time

    self._has_runner = True
    data = self.new_data
    capture = self.capture(data)
    data_time = time.time() - self.run_interval

    while True:
      self.sleep(data_time)

      with self._running_lock:
        # try to capture the next data segment
        if self._running:
          data_time = time.time()
          data = next(capture)
        else:
          self._has_runner = False
          return

      # check if the data chunk is emtpy or not
      if len(data)>0:
        # save the current data to temp
        with self._file_lock:
          if self._file is not None:
            self.save(data)

        # buffer the current data
        self.data = data[-1]

  def capture(self, data):
    while True:
      get_all = False
      data_list = []
      while not get_all and len(data_list)<FRAME_BUFFER:
        try:
          im = self._spincam.GetNextImage() #TODO: add a timeout
          if im.IsIncomplete():
            status = im.GetImageStatus()
            im.Release()
            raise Exception(f"Image incomplete with image status {status} ...")
          data = im.GetNDArray()
          data_list.append(data)
          im.Release()

        except PySpin.SpinnakerException as e:
          self.print(f'Error in spinnaker: {e}. Assumed innocuous.')
          get_all = True
          continue
      yield data_list

  def open_file(self, filepath):
    # path = os.path.join(filepath, f'{self.device_serial_number}.mp4')
    self.print(f'saving camera data to {filepath}')
    return ffmpeg \
        .input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{self.width}x{self.height}', framerate=self.run_rate) \
        .output(filepath, vcodec='libx265') \
        .overwrite_output() \
        .global_args('-loglevel', 'error') \
        .run_async(pipe_stdin=True, quiet=True)
    # .run_async(pip_stdin=True)

  def close_file(self, fileObj):
    fileObj.stdin.close()
    # fileObj.kill()
    fileObj.wait()
    self.print('done waiting for ffmpeg')
    # TODO: figure out how to gracefully close this

  def save(self, data):
    for a in data:
      self._file.stdin.write(a.tobytes())

  def get_camera_properties(self):
    nodemap_tldevice = self._spincam.GetTLDeviceNodeMap()
    device_serial_number = PySpin.CStringPtr(
        nodemap_tldevice.GetNode('DeviceSerialNumber')).GetValue()
    nodemap = self._spincam.GetNodeMap()
    height = PySpin.CIntegerPtr(nodemap.GetNode('Height')).GetValue()
    width = PySpin.CIntegerPtr(nodemap.GetNode('Width')).GetValue()
    return device_serial_number, height, width

  def predisplay(self, frame):
    # TODO: still get frame as input? but should return some kind of dictionary? or array?
    process = self.processing
    #######
    # data_count = self.data_count
    # cv2.putText(frame,str(data_count),(50, 50),cv2.FONT_HERSHEY_PLAIN,3.0,255,2)
    # print(f'sent frame {data_count}')
    #######
    if process is not None:
      results = self.results
      if results is not None:
        if process['mode'] == 'DLC':
          draw_dots(frame, results)
          cv2.putText(frame, f"frame number {process['frame_num']}", (50, 50),
                      cv2.FONT_HERSHEY_PLAIN, 4.0, (255, 0, 125), 2)
        else:
          cv2.putText(frame, f"Performing {process['mode']} calibration", (50, 50),
                      cv2.FONT_HERSHEY_PLAIN, 4.0, (255, 0, 125), 2)

          if str(self.device_serial_number) != str(TOP_CAM) and process['mode'] == 'intrinsic':
            if 'calibrator' in process.keys():
              cv2.drawChessboardCorners(
                  frame, (process['calibrator'].x, process['calibrator'].y), results['corners'], results['ret'])
          else:
            if len(results['corners']) != 0:
              cv2.aruco.drawDetectedMarkers(
                  frame, results['corners'], results['ids'], borderColor=225)

          if process['mode'] == 'alignment':
            if results['allDetected']:
              text = 'Enough corners detected! Ready to go'
            else:
              text = "Not enough corners! Please adjust the camera"

            cv2.putText(frame, text, (500, 1000),
                        cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 255), 2)
          if process['mode'] == 'extrinsic':
            if results['ids'] is None:
              text = 'Missing board or intrinsic calibration file'
              cv2.putText(frame, text, (500, 1000),
                          cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 255), 2)
            elif results['ret']:
              text = 'Frame was useable'
              cv2.putText(frame, text, (500, 1000),
                          cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 255), 2)
    return frame #gets drawn to screen


  def close(self):
    self._spincam.DeInit()

  def __del__(self):
    self._spincam.DeInit()
