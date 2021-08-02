import time
import PySpin
import os

import threading
from Camera import Camera
from Nidaq import Nidaq
from Mic import Mic

# import ProcessingGroup as pg
# import RigStatus
CAM_LIST = [17391304, 17391290, 19287342, 19412282]


def rearrange_cameras(cameras: list):
  serial_number_list = [int(camera.device_serial_number) for camera in cameras]
  zipper = dict(zip(serial_number_list, cameras))
  new_cameras = [zipper[key] for key in CAM_LIST if key in zipper.keys()]
  return new_cameras


class AcquisitionGroup:
  # def __init__(self, frame_rate=30, audio_settings=None):
  def __init__(self, status, hostname='localhost', ports=5002):
    self._system = PySpin.System.GetInstance()
    self._camlist = self._system.GetCameras()
    self.nCameras = self._camlist.GetSize()
    # +1: Nidaq. +2: Nidaq + Mic
    self.nChildren = self.nCameras + 2
    if not isinstance(ports, list):
      ports = [ports + i for i in range(self.nChildren)]

    cameras = [Camera(self, self._camlist, i, status['frame rate'].current, (hostname, ports[i]))
               for i in range(self.nCameras)]

    self.cameras = rearrange_cameras(cameras)
    self.camera_order = CAM_LIST
    self.mic = Mic(self, status['sample frequency'].current, status['spectrogram'].current, (
        hostname, ports[-2]))
    self.nidaq = Nidaq(self, status['frame rate'].current,
                       status['sample frequency'].current, status['spectrogram'].current, (
                           hostname, ports[-1]))

    self.children = self.cameras + [self.mic] + [self.nidaq]

    self._processors = [None] * self.nChildren
    self._runners = [None] * self.nChildren
    self._starters=[None]*self.nChildren
    self._displayers = [None] * self.nChildren
    self.filepaths = None
    #self._starters

    self.started = False
    self.processing = False
    self.running = False

    # self.pg = pg.ProcessingGroup()

    # self.print('done setting up ag. is camera 3 running? ',
    #           self.cameras[3].running)

  def start(self, filepaths=None, isDisplayed=True):
    if self.started:
      self.print('already started. Filepath/display unchanged.')
      return

    self.filepaths = filepaths
    if not self.filepaths:
      self.filepaths = [None] * self.nChildren
    if not isDisplayed:
      isDisplayed = [False] * self.nChildren
    elif not isinstance(isDisplayed, list) or len(isDisplayed) == 1:
      # TODO: does this work? what if isDisplayed = [True]?
      # wouldn't we then get [[True], [True], [True], ...] ?
      isDisplayed = [isDisplayed] * self.nChildren

    self.print('detected %d cameras' % self.nCameras)

    for i,child, fp, disp in zip(list(range(len(self.cameras))),self.cameras, self.filepaths[: -2], isDisplayed[: -2]):
      self._starters[i]=threading.Thread(target=child.start,kwargs={'filepath':fp,'display':disp})
      self._starters[i].start()
      #child.start(filepath=fp, display=disp)
      self.print('started camera ' + child.device_serial_number)

    # start mic
    self.mic.start(filepath=self.filepaths[-2], display=isDisplayed[-1])
    self.print('started mic')

    # once the camera BeginAcquisition methods are called, we can start triggering
    # ^ this is false, triggering happens in ag.run() ??
    self.nidaq.start(filepath=self.filepaths[-1], display=False)
    self.print('started nidaq')

    self.started = True

  def run(self):
    # begin gathering samples
    # if not self._runners:  # if self._runners == []
    #   for i, child in enumerate(self.children):
    #     self._runners.append(threading.Thread(target=child.run))
    #     self._runners[i].start()

    # self._runners.append(threading.Thread(target=self.nidaq.run))
    # self._runners[-1].start()

    # else:
    for i, child in enumerate(self.children):
      if self._runners[i] is None or not self._runners[i].is_alive():
        self._runners[i] = threading.Thread(target=child.run)
        #threading.setprofile(child.run) # for profiler
        print('start running child %d'%i)
        self._runners[i].start()
      if self._displayers[i] is None or not self._displayers[i].is_alive():
        #if i != 4:  # temporaray solution for not displaying nidaq spectrogram
          self._displayers[i] = threading.Thread(target=child.display)
          #threading.setprofile(child.display)# for profiler
          self._displayers[i].start()
          print('start displaying child %d' % i)
    self.running = True

    self.print('finished AcquisitionGroup.run')

  def process(self, i, options):
    # if it's recording, process() shouldn't be run. except dlc
    if not all(self.filepaths) or options['mode'] == 'DLC':
      if self._processors[i] is None or not self._processors[i].is_alive():

        # turn on top camera processing

        if options['mode'] == 'extrinsic':
          # turn on all cameras
          for j, camera in enumerate(self.cameras):
            camera.processing = options
            self._processors[j] = threading.Thread(
                target=camera.run_processing)
            self._processors[j].start()
        else:
          self.children[i].processing = options
          self._processors[i] = threading.Thread(
              target=self.children[i].run_processing)
          self._processors[i].start()
    self.processing = True

  # TODO: this should be refined depending on future changes to processing
  def stop_processing(self, i):
    self.children[i].processing = None
    self._processors[i].join()

  def stop(self):

    for i,cam in enumerate(self.cameras):
      self._starters[i].join()
      cam.stop()
    self.mic.stop()
    self.nidaq.stop()
    for child in self.children:
      self.print('waiting for next child')
      child.wait_for()
    # del self.children
    # self._processors = [None] * self.nChildren

    self.processing = False
    self.running = False
    self.started = False

    self.print('finished AcquisitionGroup.stop()')

  def restart(self, filepaths=None, isDisplayed=True):
    self.stop()
    self.start(filepaths, isDisplayed)
    self.run()

  def print(self, *args):
    print(*args)

  # def update(self, rootfilename, stepnumber):
  #   #send a message to the gui that a step ahs been completed

  def __del__(self):
    del self.children
    self._camlist.Clear()
    self._system.ReleaseInstance()
    # del self.nidaq
