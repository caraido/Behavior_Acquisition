from utils.audio_processing import read_audio
import scipy.io.wavfile as wavfile

path = r'D:\Desktop\2021-06-04_p6_pups_and_mama\B&K_audio.tdms'

audio,_=read_audio(path)
wavfile.write(path[:-4]+'wav', int(2.5e5), audio)

