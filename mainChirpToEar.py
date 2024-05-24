import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
from UserModules.pyDPOAEmodule import sendChirpToEar, getSClat

fsamp = 44100 # sampling rate
MicGain = 40 # gain of the microphone

AmpChirp = 0.04   # amplitude of the chirp signal
buffersize = 2048
lat_SC = getSClat(fsamp,buffersize)

Hinear1, Hinear2, fxinear, y1, y2 = sendChirpToEar(AmpChirp=AmpChirp,fsamp=fsamp,MicGain=MicGain,Nchirps=300,buffersize=buffersize,latency_SC=lat_SC)


pathfolder = 'Calibration_files/Files'
if not os.path.exists(pathfolder):
    os.makedirs(pathfolder)
filename = 'InEarCalData'
caldata = {'Hinear1':Hinear1,'Hinear2':Hinear2,'fxinear':fxinear,'AmpChirp':AmpChirp,'lat_SC':lat_SC}

savemat(pathfolder + '/' + filename + '.mat', caldata)


fig,ax1 = plt.subplots()
ax1.semilogx(fxinear/1e3,np.abs(Hinear1))
ax1.semilogx(fxinear/1e3,np.abs(Hinear2))
ax1.set_xlabel('Frequency (kHz)')
ax1.set_ylabel('|H(f)| (-)')
ax1.set_xlim((0.1, 13))
ax1.set_title('Absolute value of the transfer function for each speaker')
ax1.legend(('Speaker 1','Speaker 2'))

plt.show()
