#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:11:08 2023

@author: audiobunka
"""


from scipy.io import loadmat
import os
from UserModules.pyUtilities import butter_highpass_filter 
import numpy as np
import matplotlib.pyplot as plt

import UserModules.pyDPOAEmodule as pDP
    


path = "Results/s083/"
subj_name = 's083'

# left ear
# p4swDPOAE_s083_24_03_26_14_45_37_F2b_8000HzF2a_500HzL1_65dB_L2_65dB_f2f1_120.0_Oct_20_L_01 # file name
ear = 'left'
deL = '24_03_26_14_45_37' # sequence of date and time from the result file to extract all files for one session



f2f1 = 1.2

fsamp = 96000;


def getDPgram(path,DatePat,fsamp,r):

    dir_list = os.listdir(path)
    
    n = 0
    for k in range(len(dir_list)):
        if  DatePat in dir_list[k]:
            data = loadmat(path+ dir_list[k])
            lat = data['lat_SC'][0][0]
            print([path+ dir_list[k]])
            print(f"SC latency: {lat}")
            recSig1 = data['recsigp1'][:,0]
            recSig2 = data['recsigp2'][:,0]
            recSig3 = data['recsigp3'][:,0]
            recSig4 = data['recsigp4'][:,0]
            #print(np.shape(recSig))
            recSig1 = butter_highpass_filter(recSig1, 200, fsamp, order=5)
            recSig1 = np.reshape(recSig1,(-1,1)) # reshape to a matrix with 1 column
            recSig2 = butter_highpass_filter(recSig2, 200, fsamp, order=5)
            recSig2 = np.reshape(recSig2,(-1,1)) # reshape to a matrix with 1 column
            recSig3 = butter_highpass_filter(recSig3, 200, fsamp, order=5)
            recSig3 = np.reshape(recSig3,(-1,1)) # reshape to a matrix with 1 column
            recSig4 = butter_highpass_filter(recSig4, 200, fsamp, order=5)
            recSig4 = np.reshape(recSig4,(-1,1)) # reshape to a matrix with 1 column
            if n == 0:
                recMat1 = recSig1[lat:,0]
                recMat2 = recSig2[lat:,0]
                recMat3 = recSig3[lat:,0]
                recMat4 = recSig4[lat:,0]
            else:
                recMat1 = np.c_[recMat1, recSig1[lat:,0]]  # add to make a matrix with columns for every run
                recMat2 = np.c_[recMat2, recSig2[lat:,0]]  # add to make a matrix with columns for every run
                recMat3 = np.c_[recMat3, recSig3[lat:,0]]  # add to make a matrix with columns for every run
                recMat4 = np.c_[recMat4, recSig4[lat:,0]]  # add to make a matrix with columns for every run
            n += 1
        
    
    
    
    recMean1 = np.median(recMat1,1)  # median across rows
    recMean1 = np.reshape(recMean1,(-1,1))
    recMean2 = np.median(recMat2,1)  # median across rows
    recMean2 = np.reshape(recMean2,(-1,1))
    recMean3 = np.median(recMat3,1)  # median across rows
    recMean3 = np.reshape(recMean3,(-1,1))
    recMean4 = np.median(recMat4,1)  # median across rows
    recMean4 = np.reshape(recMean4,(-1,1))
    
    # 2. calculate noise matrix
    noiseM1 = recMat1 - recMean1
    noiseM2 = recMat2 - recMean2
    noiseM3 = recMat3 - recMean3
    noiseM4 = recMat4 - recMean4
    Nsamp = len(recMean1) # number of samples
    
    
    sigma1 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM1**2,1)))
    sigma2 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM2**2,1)))
    sigma3 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM3**2,1)))
    sigma4 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM4**2,1)))
    Nt = 8
    Theta1 = Nt*sigma1 # estimation of the threshold for sample removal
    Theta2 = Nt*sigma2 # estimation of the threshold for sample removal
    Theta3 = Nt*sigma3 # estimation of the threshold for sample removal
    Theta4 = Nt*sigma4 # estimation of the threshold for sample removal
    
    recMat1[np.abs(noiseM1)>Theta1] = np.nan
    recMat1[np.abs(noiseM2)>Theta2] = np.nan
    recMat1[np.abs(noiseM3)>Theta3] = np.nan
    recMat1[np.abs(noiseM4)>Theta4] = np.nan
    
    recMat2[np.abs(noiseM1)>Theta1] = np.nan
    recMat2[np.abs(noiseM2)>Theta2] = np.nan
    recMat2[np.abs(noiseM3)>Theta3] = np.nan
    recMat2[np.abs(noiseM4)>Theta4] = np.nan
    
    recMat3[np.abs(noiseM1)>Theta1] = np.nan
    recMat3[np.abs(noiseM2)>Theta2] = np.nan
    recMat3[np.abs(noiseM3)>Theta3] = np.nan
    recMat3[np.abs(noiseM4)>Theta4] = np.nan
    
    recMat4[np.abs(noiseM1)>Theta1] = np.nan
    recMat4[np.abs(noiseM2)>Theta2] = np.nan
    recMat4[np.abs(noiseM3)>Theta3] = np.nan
    recMat4[np.abs(noiseM4)>Theta4] = np.nan
    
    noiseM1[np.abs(noiseM1)>Theta1] = np.nan
    noiseM1[np.abs(noiseM2)>Theta2] = np.nan
    noiseM1[np.abs(noiseM3)>Theta3] = np.nan
    noiseM1[np.abs(noiseM4)>Theta4] = np.nan
    
    noiseM2[np.abs(noiseM1)>Theta1] = np.nan
    noiseM2[np.abs(noiseM2)>Theta2] = np.nan
    noiseM2[np.abs(noiseM3)>Theta3] = np.nan
    noiseM2[np.abs(noiseM4)>Theta4] = np.nan
    
    noiseM3[np.abs(noiseM1)>Theta1] = np.nan
    noiseM3[np.abs(noiseM2)>Theta2] = np.nan
    noiseM3[np.abs(noiseM3)>Theta3] = np.nan
    noiseM3[np.abs(noiseM4)>Theta4] = np.nan
    
    noiseM4[np.abs(noiseM1)>Theta1] = np.nan
    noiseM4[np.abs(noiseM2)>Theta2] = np.nan
    noiseM4[np.abs(noiseM3)>Theta3] = np.nan
    noiseM4[np.abs(noiseM4)>Theta4] = np.nan
    
    oaeDS = (np.nanmean(recMat1,1)+np.nanmean(recMat2,1)+np.nanmean(recMat3,1)+np.nanmean(recMat4,1))/4  # exclude samples set to NaN (noisy samples)
    nfloorDS = (np.nanmean(noiseM1,1)+np.nanmean(noiseM2,1)+np.nanmean(noiseM3,1)+np.nanmean(noiseM4,1))/4  # exclude samples set to NaN (noisy samples)
    
    #nfloorDS = np.nanmean(noiseM,1)
    #from scipy.io import savemat
    #data2 = {'oaeDS':oaeDS}
    #savemat('oaeDS015python.mat',data2)
    
    
    f2f1 = 1.2
    f2b = 8000
    f2e = 500
    
    octpersec = r
    GainMic = 40
    rF = 1.2
    #(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tau02,tshift):
    
    if f2b<f2e:
        T = np.log2(f2e/f2b)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        
    else:
        T = np.log2(f2b/f2e)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        
    L1sw =  T/np.log(f2e/f2b) 
    #pDP.fceDPOAEinwinSS(oaeDS,Nsamp,f2b/rF,L1sw,rF,fsamp,0.01,0.02,0)
    #oaeDS = np.concatenate((oaeDS[:int(T*fsamp)],np.zeros((Nsamp4-int(T*fsamp),))))
    #hmfftlen = 2**14
    
    DPgram, DPgramNL, DPgramCR, NF, NFnl, fx = pDP.calcDPgramFAV_HS(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,GainMic)
    
    
    
    return DPgram, DPgramNL, DPgramCR, NF, NFnl, fx

r = 2 # sweep rate (oct/sec)
DPgram, DPgramNL, DPgramCR, NF, NFnl, fx = getDPgram(path,deL,fsamp ,r)

f2x = f2f1*fx/(2-f2f1)

f2min = 500
f2max = 8000
fig,(ax1,ax2) = plt.subplots(2,1)

LWv = np.linspace(1,1,9)

cycle = np.pi*2 
#ax1.plot(fx,20*np.log10(np.abs(DPgram[:,-1])/(np.sqrt(2)*2e-5)))
ax1.plot(f2x,20*np.log10(np.abs(DPgram)/(np.sqrt(2)*2e-5)),color='C0',linewidth=LWv[0])
ax1.plot(f2x,20*np.log10(np.abs(DPgramNL)/(np.sqrt(2)*2e-5)),color='C1',linewidth=LWv[0])
ax1.plot(f2x,20*np.log10(np.abs(DPgramCR)/(np.sqrt(2)*2e-5)),color='C2',linewidth=LWv[0])
ax1.plot(f2x,20*np.log10(np.abs(NF)/(np.sqrt(2)*2e-5)),linestyle=':',color='C0',linewidth=LWv[0])
ax1.plot(f2x,20*np.log10(np.abs(NFnl)/(np.sqrt(2)*2e-5)),linestyle=':',color='C1',linewidth=LWv[0])

ax1.set_xlim([f2min,f2max])
ax1.set_ylim([-40,30])
ax1.set_ylabel('Amplitude (dB SPL)')
ax1.set_title('DP-gram for scissor paradigm, '+ subj_name + ', ' + ear + ' ear')
ax1.legend(('DPOAE','DPOAE SL','DPOAE LL','N. floor DPOAE','N. floor DPOAE SL'))
#ax1.legend(('DPOAE','NL comp.','CR comp.','noise floor'))
DPphaseU = np.copy(np.angle(DPgram)[~np.isnan(DPgram)])
DPphaseU = np.unwrap(DPphaseU)

DPphaseNLU = np.copy(np.angle(DPgramNL)[~np.isnan(DPgramNL)])
DPphaseNLU = np.unwrap(DPphaseNLU)

DPphaseCRU = np.copy(np.angle(DPgramCR)[~np.isnan(DPgramCR)])
DPphaseCRU = np.unwrap((DPphaseCRU))


ax2.plot(f2x[~np.isnan(DPgram)],DPphaseU/cycle,color='C0',linewidth=LWv[1])
ax2.plot(f2x[~np.isnan(DPgramNL)],DPphaseNLU/cycle,color='C1',linewidth=LWv[1])
ax2.plot(f2x[~np.isnan(DPgram)],DPphaseCRU/cycle,color='C2',linewidth=LWv[1])


#ax2.plot(fx,np.unwrap(np.angle(DPgramNL))/cycle)
#ax2.plot(fx,np.unwrap(np.angle(DPgramCR))/cycle)
ax2.set_xlim([f2min,f2max])
ax2.set_ylim([-40,10])
ax2.set_xlabel('Frequency $f_2$ (Hz)')
ax2.set_ylabel('Phase (cycles)')
plt.show()