# this is the place to put all kinds of settings

# dlc settings
DLC_LIVE_MODEL_PATH=r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\DLC\Alec_second_try-Devon-2020-12-07\exported-models\DLC_Alec_second_try_resnet_50_iteration-0_shuffle-1'
number_of_camera=4


# camera settings
FRAME_TIMEOUT = 10  # time in milliseconds to wait for pyspin to retrieve the frame
FRAME_BUFFER = 3 # frames buffer for display and save
DLC_RESIZE = 0.6  # resize the frame by this factor for DLC
DLC_UPDATE_EACH = 3  # frame interval for DLC update
TOP_CAM='17391304'
TEMP_PATH = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\config'
N_BUFFER=2000

# triangulation settings
TRI_THRESHOLD=0.7

# calibration settings
CALIB_UPDATE_EACH = 1  # frame interval for calibration frame update

# kalman filter settings
dt = 1 / 30
CUTOFF = 0.7
DISTRUSTNESS = 1e22

# gemometry settings
FRAME_RATE=15

# imag draw settings
TOP_THRESHOLD=0.8

# path operation settings
saving_path_prefix = 'D:\\'
default_saving_path= 'Desktop'
default_folder_name = 'Testing'
GLOBAL_CONFIG_PATH = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\config'
GLOBAL_CONFIG_ARCHIVE_PATH = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\config_archive'
global_log_path=r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\log'
namespace_path = r'C:\Users\SchwartzLab\PycharmProjects\bahavior_rig\behavior_gui\assets\namespace\namespace.json'

# squeaks settings
UTILS_PATH=r'C:\\Users\\SchwartzLab\\PycharmProjects\\bahavior_rig\\utils'
DEEPSQUEAK_PATH=r"C:\\Users\\SchwartzLab\\MatlabProjects"
default_network='All Short Calls_Network_V1.mat'
SETTINGS=[0,8,0.002,80,35,0,1]

