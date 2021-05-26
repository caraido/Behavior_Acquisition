# from setup import ag, status
from utils import path_operation_utils as pop


# serialNumbers=[19287342,19412282,17391304,17391290]

def initCallbacks(ag, status):

  def initialization(state):
    if state == 'initialized':
      ag.start()  # need the filepaths here for the temp videos?
      ag.run()
    else:
      ag.stop()

  status['initialization'].callback(initialization)  # TODO: decorators?

  def recording(state):
    ag.print(f'got to recording callback with state == {state}')
    if state:
      rootfilename = status['rootfilename'].current
      ag.print(f'rootfilename was: {rootfilename}')
      camera_list = []
      for i in range(ag.nCameras):
        camera_list.append(ag.cameras[i].device_serial_number)
      filepaths = pop.reformat_filepath('', rootfilename, camera_list)

      ag.stop()
      ag.start(filepaths=filepaths)

      ag.run()
      status['initialization'].immutable()
      status['calibration'].immutable()
      # TODO: make rootfilename and notes immutable here? and mutable below? for safety
    else:
      ag.print('got stop message')
      ag.stop()
      ag.start()  # restart without saving
      ag.run()

      status['initialization'].mutable()
      status['calibration'].mutable()
      status['rootfilename']('')  # to make sure we don't accidentally

  status['recording'].callback(recording)

  def rootfilename(state):
    # just temporary, for debugging. want to make sure order is consistent.
    #
    ag.print(f'attempted to set rootfilename to "{state}"')

  status['rootfilename'].callback(rootfilename)

  def notes(state):
    # TODO: should save notes under rootfile
    # status['rootfilename'].current
    ag.print(f'attempted to update notes')
    ag.print(state)
    ag.print(ag.filepaths)
    pop.save_notes(state, ag.filepaths)

  status['notes'].callback(notes)

  def calibration(state):

    if state['is calibrating']:
      # TODO: start calibrating in background thread
      # state['camera serial number'].current #gives the current camera SN
      # state['type'].current == 'Intrinsic' #intrinsice or extrinsic?

      # ag.stop()
      # ag.start(with_filepaths) #<-- key line
      # ag.run()

      cam_id = state['camera serial number']
      cam_num = ag.camera_order.index(cam_id)
      type = state['calibration type']
      process = {'mode': type}
      # extrinsic calibration will trigger saving temp video to config path
      if type == 'extrinsic':
        camera_list = []
        for i in range(ag.nCameras):
          camera_list.append(ag.cameras[i].device_serial_number)
        config_filepaths = pop.get_extrinsic_path(camera=camera_list)
        ag.stop()
        ag.start(filepaths=config_filepaths)
        ag.process(cam_num, options=process)
        # ag.run()
      else:
        ag.process(cam_num, options=process)

      status['initialization'].immutable()

    else:
      ag.stop()
      ag.start()
      ag.run()
      status['initialization'].mutable()

  status['calibration'].callback(calibration)

  def spectrogram(state):
    ag.print(f'applying new status from state: {state}')
    ag.mic.parse_settings(status['spectrogram'].current)
    # TODO: trying to update _nx or _nfft will cause an error
    # that means we can only update log scaling and noise correction

    # TODO: update the port number... if _nx or _nfft change

  status['spectrogram'].callback(spectrogram)

  # TODO: following should be refined to handle different analyses types, modes, etc.
  def analyze(state):
    ag.print(f'toggling analysis')
    if state:
      ag.process(0, {'mode': 'DLC'})
    else:
      ag.stop_processing(0)  # stops processing without stopping acquisition

    status['analyzing'].callback(analyze)

  # def camera(state):
  #cameraId = state['camera index'].current
  # if state['serial number'].current != status[f'camera {cameraId}'].current['serial number'].current:
  #  temp1 = status[f'camera {cameraId].current.copy()
  #  temp2 = status[f'camera {camera where serialNumber == requested...

  # status[f'camera {cameraId}].current = temp2
  # ...

  # for i in range(4):
  #  status[f'camera {i}'].callback(camera)
