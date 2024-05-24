import numpy as np
from scipy.io import savemat, loadmat

import datetime
from UserModules.pyDPOAEmodule import RMEplayrec,  generateDPOAEstimulusPhase, calcDPgramFAV_HS, getSClat
from UserModules.pyUtilities import butter_highpass_filter
import matplotlib.pyplot as plt
# parameters of evoking stimuli

f2b = 8000  # f1 start frequency
f2e = 500 # f2 end frequency
f2f1 = 1.2  # f2/f1 ratio
L2 = 45    # intensity of f2 tone
L1 = int(0.4*L2+39)  # scissor paradigm

r = 2   # sweep rate in octaves per second
fsamp = 96000; bufsize = 4096

lat_SC = getSClat(fsamp,bufsize)
micGain = 40  # gain in the OAE probe
ear_t = 'L' # which ear is recorded, L for left, R for right

plt.close('all')

save_path = 'Results/s039/'  # path where the recorded responses are saved
subj_name = 's039'  # subject name


def get_time() -> str:
    # to get current time
    now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return now_time


# data initialization
    
# generate stimuli using the phase ensemble such that 4 reseponses are saved to keep only 2f1-f2 component
s1p1,s2p1 = generateDPOAEstimulusPhase(f2f1, fsamp, f2b, f2e, 0,0, r, L1, L2)    

s1p2,s2p2 = generateDPOAEstimulusPhase(f2f1, fsamp, f2b, f2e, np.pi/2, np.pi, r, L1, L2)    

s1p3,s2p3 = generateDPOAEstimulusPhase(f2f1, fsamp, f2b, f2e, np.pi, 2*np.pi, r, L1, L2)    

s1p4,s2p4 = generateDPOAEstimulusPhase(f2f1, fsamp, f2b, f2e, 3*np.pi/2, 3*np.pi, r, L1, L2)    

# based on data acquisition approach, make a matrix or not?        
sp1 = np.vstack([s1p1,s2p1,s1p1+s2p1]).T  # make matrix where the first column
sp2 = np.vstack([s1p2,s2p2,s1p2+s2p2]).T  # make matrix where the first column
sp3 = np.vstack([s1p3,s2p3,s1p3+s2p3]).T  # make matrix where the first column
sp4 = np.vstack([s1p4,s2p4,s1p4+s2p4]).T  # make matrix where the first column

# the third channel is connected with input, can be neglected

if f2e>f2b:
    numofoct = np.log2(f2e/f2b)  # number of octaves for upward sweep
else:
    numofoct = np.log2(f2b/f2e)  # number of octaves for downward sweep

T = numofoct/r   # time duration of the sweep for the given sweep rate
        
t = get_time() # current date and time to save it into the file name


# load calibration data and save them to results
calib_data = loadmat('Calibration_files/Files/InEarCalData.mat')

file_name = 'calib_data_' + subj_name + '_' + t[2:] + '_' + ear_t
savemat(save_path + '/' + file_name + '.mat', calib_data)

# measurement phase
counter = 0

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)



try:
    while True:
        counter += 1
        print('Rep: {}'.format(counter))    
        recsigp1 = RMEplayrec(sp1,fsamp,SC=21,buffersize=bufsize) # send signal to the sound card
        #time.sleep(1) 
        recsigp2 = RMEplayrec(sp2,fsamp,SC=21,buffersize=bufsize) # send signal to the sound card
        #time.sleep(1) 
        recsigp3 = RMEplayrec(sp3,fsamp,SC=21,buffersize=bufsize) # send signal to the sound card
        #time.sleep(1) 
        recsigp4 = RMEplayrec(sp4,fsamp,SC=21,buffersize=bufsize) # send signal to the sound card
        if counter<10:  # to add 0 before the counter number
            counterSTR = '0' + str(counter)
        else:
            counterSTR = str(counter)    
        # every recorded response is saved, so first make a dictionary with needed data
        data = {"recsigp1": recsigp1,"recsigp2": recsigp2,"recsigp3": recsigp3,"recsigp4": recsigp4,"fsamp":fsamp,"f2f1":f2f1,"f2b":f2b,"f2e":f2e,"r":r,"L1":L1,"L2":L2,"lat_SC":lat_SC}  # dictionary
        file_name = 'p4swDPOAE_' + subj_name + '_' + t[2:] + '_' + 'F2b' + '_' + str(f2b) + 'HzF2a' + '_' + str(f2e) + 'HzL1' + '_' + str(L1) + 'dB' + '_' + 'L2' + '_' + str(L2) + 'dB' + '_' + 'f2f1' + '_' + str(f2f1 * 100) + '_' + 'Oct' + '_' + str(r * 10) + '_' + ear_t + '_' + counterSTR
        savemat(save_path + '/' + file_name + '.mat', data)

        # now do processing to show the result to the experimenter
        #    
        cut_off = 200 # cut of frequency of the high pass filter
        recSigp1 = butter_highpass_filter(recsigp1[:,0], cut_off, fsamp, order=5)

        recsigp1 = np.reshape(recSigp1,(-1,1)) # reshape to a matrix with 1 column

        recSigp2 = butter_highpass_filter(recsigp2[:,0], cut_off, fsamp, order=5)

        recsigp2 = np.reshape(recSigp2,(-1,1)) # reshape to a matrix with 1 column

        recSigp3 = butter_highpass_filter(recsigp3[:,0], cut_off, fsamp, order=5)

        recsigp3 = np.reshape(recSigp3,(-1,1)) # reshape to a matrix with 1 column

        recSigp4 = butter_highpass_filter(recsigp4[:,0], cut_off, fsamp, order=5)

        recsigp4 = np.reshape(recSigp4,(-1,1)) # reshape to a matrix with 1 column

        #recsig = (recsigp1 + recsigp2 + recsigp3 + recsigp4)/4

        

        if counter == 1:
            ricMat1 = recsigp1[lat_SC:,0]
            ricMat2 = recsigp2[lat_SC:,0]
            ricMat3 = recsigp3[lat_SC:,0]
            ricMat4 = recsigp4[lat_SC:,0]
            
        elif counter > 1 and counter < 3:
            ricMat1 = np.c_[ricMat1, recsigp1[lat_SC:,0]]
            ricMat2 = np.c_[ricMat2, recsigp2[lat_SC:,0]]
            ricMat3 = np.c_[ricMat3, recsigp3[lat_SC:,0]]
            ricMat4 = np.c_[ricMat4, recsigp4[lat_SC:,0]]
            
        else:
            ricMat1 = np.c_[ricMat1, recsigp1[lat_SC:,0]]
            ricMat2 = np.c_[ricMat2, recsigp2[lat_SC:,0]]
            ricMat3 = np.c_[ricMat3, recsigp3[lat_SC:,0]]                
            ricMat4 = np.c_[ricMat4, recsigp4[lat_SC:,0]]                
                

            recMean1 = np.median(ricMat1,1)
            recMean2 = np.median(ricMat2,1)
            recMean3 = np.median(ricMat3,1)
            recMean4 = np.median(ricMat4,1)

            recMean1 = np.reshape(recMean1, (-1,1))
            recMean2 = np.reshape(recMean2, (-1,1))
            recMean3 = np.reshape(recMean3, (-1,1))
            recMean4 = np.reshape(recMean4, (-1,1))
            recMat1 = np.array(ricMat1)
            recMat2 = np.array(ricMat2)
            recMat3 = np.array(ricMat3)
            recMat4 = np.array(ricMat4)
            
            noiseM1 = recMat1 - recMean1
            noiseM2 = recMat2 - recMean2
            noiseM3 = recMat3 - recMean3
            noiseM4 = recMat4 - recMean4

            Nsamp = len(recMean1) # number of samples
            print(Nsamp)

            sigma1 = np.sqrt(1/(Nsamp*counter)*np.sum(np.sum(noiseM1**2,1)))
            sigma2 = np.sqrt(1/(Nsamp*counter)*np.sum(np.sum(noiseM2**2,1)))
            sigma3 = np.sqrt(1/(Nsamp*counter)*np.sum(np.sum(noiseM3**2,1)))
            sigma4 = np.sqrt(1/(Nsamp*counter)*np.sum(np.sum(noiseM4**2,1)))
            Theta1 = 8*sigma1 # estimation of the threshold for sample removal
            Theta2 = 8*sigma2 # estimation of the threshold for sample removal
            Theta3 = 8*sigma3 # estimation of the threshold for sample removal
            Theta4 = 8*sigma4 # estimation of the threshold for sample removal

            # now we have to make NAN where the noise was to large, but we have to do it in the wave
            # which 
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


            oaeDS = (np.nanmean(recMat1,1) + np.nanmean(recMat2,1) + np.nanmean(recMat3,1) + np.nanmean(recMat4,1))/4  # exclude samples set to NaN (noisy samples)
            nfloorDS = (np.nanmean(noiseM1,1) + np.nanmean(noiseM2,1) + np.nanmean(noiseM3,1) + np.nanmean(noiseM4,1))/4
            
            #print(oaeDS)
            #calculate frequency response
            [DPOAEcalib, DPOAEcalibNL, DPOAEcalibCR, NFLOORcalib, NFLOORcalibNL, fxI] = calcDPgramFAV_HS(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,r,micGain)
            

            ax1.clear()
            ax2.clear()                
            fx2 = f2f1*fxI/(2-f2f1)
            ax1.plot(fx2,20*np.log10(abs(DPOAEcalib)/(np.sqrt(2)*2e-5)))
            ax1.plot(fx2,20*np.log10(abs(DPOAEcalibNL)/(np.sqrt(2)*2e-5)))
            ax1.plot(fx2,20*np.log10(abs(DPOAEcalibCR)/(np.sqrt(2)*2e-5)))
            ax1.plot(fx2,20*np.log10(abs(NFLOORcalib)/(np.sqrt(2)*2e-5)),':')
            ax1.plot(fx2,20*np.log10(abs(NFLOORcalibNL)/(np.sqrt(2)*2e-5)),':')
            ax1.set_ylim((-40,30))
            ax1.set_ylabel('Amplitude (dB SPL)')
            ax1.legend(('DPOAE','NL comp.','CR comp.','noise floor'))
            
            cycle = 2*np.pi
            ax2.plot(fx2,np.unwrap(np.angle(DPOAEcalib))/cycle)
            ax2.plot(fx2,np.unwrap(np.angle(DPOAEcalibNL))/cycle)
            ax2.plot(fx2,np.unwrap(np.angle(DPOAEcalibCR))/cycle)
            ax2.set_ylim((-40,2))
            ax2.set_xlabel('Frequency $f_{dp} (Hz)$')
            ax2.set_ylabel('Phase (cycles)')

            if f2b<f2e:
                ax1.set_xlim((f2b,f2e))
                ax2.set_xlim((f2b,f2e))
            else:
                ax1.set_xlim((f2e,f2b))            
                ax2.set_xlim((f2e,f2b))
            
            
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.0001) #Note this correction
            
    
except KeyboardInterrupt:
    plt.show()
    pass
            


        
    
