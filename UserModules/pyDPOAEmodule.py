#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:40:54 2024

@author: vencov
"""
# a modle with functions and utilities used for DPOAE measurement


import matplotlib.pylab as plt
import numpy as np
import scipy.signal as signal
import sounddevice as sd
from UserModules.pyRMEsd import choose_soundcard, RMEplayrec
from UserModules.pyUtilities import butter_highpass_filter, rfft
from scipy.io import loadmat
from scipy.signal import savgol_filter  # import savitzky golay filter
from scipy.signal.windows import blackman, tukey
import time



def generateSSSPhase(fs,fstart,fstop,phase,octpersec,Level,channel):
    '''
    generate synchronized swept sine with chosen phase
    '''
    if fstop>fstart:
        numofoct = np.log2(fstop/fstart)
    else:
        numofoct = np.log2(fstart/fstop)

    T = numofoct/octpersec         # time duration of the sweep
    #print(T)
    
    #fade = [441, 441]   # samlpes to fade-in and fade-out the input signal
    fade = [9600, 9600]   # samlpes to fade-in and fade-out the input signal
    L = T/np.log(fstop/fstart)
    t = np.arange(0,np.round(fs*T-1)/fs,1/fs)  # time axis
    s = np.sin(2*np.pi*(fstart)*L*np.exp(t/L)+phase)       # generated swept-sine signal

    #p0 = 2e-5
    #s = p0*10**(Level/20)*s/np.sqrt(np.mean(s**2))
    # fade-in the input signal
    if fade[0]>0:
        s[0:fade[0]] = s[0:fade[0]] * ((-np.cos(np.arange(fade[0])/fade[0]*np.pi)+1) / 2)

    # fade-out the input signal
    if fade[1]>0:
        s[-fade[1]:] = s[-fade[1]:] *  ((np.cos(np.arange(fade[1])/fade[1]*np.pi)+1) / 2)

    #s = np.pad(s, (0, 8192), 'constant')  # append zeros
    s = np.pad(s, (0, 16384), 'constant')  # append zeros

    fft_len = len(s) # number of samples for fft


    probecal = loadmat('Calibration_files/Files/InEarCalData.mat')
    if channel==1:
        Hprobe = probecal['Hinear1']
        fxprobe = probecal['fxinear']
    elif channel==2:
        Hprobe = probecal['Hinear2']
        fxprobe = probecal['fxinear']

    Sin = np.fft.rfft(s,fft_len)
    axe_w = np.linspace(0, np.pi, num=int(fft_len/2+1))
    fxSin = axe_w/(np.pi)*fs/2
    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Sin))
    #draw_DPOAE(fxSin, Sin)

    Hrint = np.interp(fxSin,fxprobe.flatten(),Hprobe.flatten())

    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Hrint))
    #draw_DPOAE(fxSin, Hrint) 

    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Sin/Hrint))
    #draw_DPOAE(fxSin, Sin/Hrint)

    SinC = Sin/Hrint
 
    sig = np.fft.irfft(SinC)

    s = 10**(Level/20)*np.sqrt(2)*2e-5*sig
    print('max value of signal: ',max(abs(s)))
    '''
    chan_in = 1 # channel of the sound card for recording
    chan_out = 3  # channel of the sound card for playing
    y = sd.playrec(s, samplerate=fs, channels=1, input_mapping=chan_in, output_mapping=chan_out, blocking=True)

    y = y.flatten()'''
    return s


def generateDPOAEstimulusPhase(f2f1,fs,f2start,f2stop,phase1,phase2,octpersec,L1,L2):
    '''
    generate stimulus to evoke DPOAEs (two synchronized swept sines)
    you can define initial phases of the tones
    '''
    f1start = f2start/f2f1
    f1stop = f2stop/f2f1

    tone1 = generateSSSPhase(fs,f1start,f1stop,phase1,octpersec,L1,1)

    tone2 = generateSSSPhase(fs,f2start,f2stop,phase2,octpersec,L2,2)

    
    return tone1, tone2  # returning back two tones separately for presentation by each speaker



def makeChirp(f1,f2,Nsamp,fsamp):
    '''
    generate a short chirp (linearly swept sine)
    '''
    T = Nsamp/fsamp  # duration of the chirp
    
    tx = np.arange(0,Nsamp/fsamp,1/fsamp)  # time axis

    A = 1
    T = Nsamp/fsamp
    
    y = np.zeros_like(tx)
    for k in range(len(tx)):
        y[k] = A*np.sin(2*np.pi*(f1+(f2-f1)/(2*(T))*tx[k])*tx[k])

    return y


def makeChirpTrain(f1,f2,Nsamp,fsamp,Nchirps):
    '''
    generate a train of chirps (linearly swept sines)
    '''

    chirp = makeChirp(f1,f2,Nsamp,fsamp) # make one chirp 
    chirptrain = np.tile(chirp,(Nchirps,))  # repeate the chirp Nchirp times
    
    return chirptrain, chirp




def makeSynchSweptSineTrain(f1,f2,T,fs,Nchirps,*,fade=(4410,4410)):
    '''
    generate a train of synch. swept sines
    '''

    chirp = makeSynchSweptSine(f1,f2,T,fs,fade=fade) # make one chirp 
    chirptrain = np.tile(chirp,(Nchirps,))  # repeate the chirp Nchirp times
    
    return chirptrain


def makeTwoPureTones(ftone1,ftone2,Ltone1,Ltone2,phitone1,phitone2,T,fs,ch1,ch2,fade=(4410,4410)):

    # tone 1    
    t1 =  makePureTone(ftone1,Ltone1,phitone1,T,fs,ch1)
    # tone 2
    t2 =  makePureTone(ftone2,Ltone2,phitone2,T,fs,ch2)
    
    if ch1==1 and ch2 == 2:
        twtones = np.column_stack((t1, t2, t1+t2))
    elif ch1==2 and ch2==1:
        twtones = np.column_stack((t2, t1, t1+t2))
    else:      
        raise ValueError("wrong cahnnels!!!")

    return twtones


           
def sendChirpToEar(*,AmpChirp=0.01,fsamp=44100,MicGain=40,Nchirps=300,buffersize=2048,latency_SC=8236):
    '''
    creates a chirptrain and sends it into the sound card
    recorded response is used for calibration of OAE probe
    Input parameters:
    AmpChirp .... Signal amplitude
    fsamp .... sampling frequency (constructed for 44100 Hz)
    MicGain .... gain on the OAE probe
    Nchirps .... number of chirps in the chirptrain
    buffersize .... number of samples in the chirp (buffer)
    latency_SC .... sound card latency
    '''
    
    #current = os.path.dirname(os.path.realpath(__file__)) # find path to the current file
    #UMdir = os.path.dirname(current)+'/UserModules'  # set path to the UserModules folder
    #sys.path.append(UMdir) # add the path to the module into sys.path
     # import needed functions from pyRMEsd module
    #from pyDPOAEmodule import makeChirpTrain  # import needed functions from pyRMEsd module

    #fsamp = 44100
    plotflag = 0  # plot responses? for debuging purposes set to 1
    f1 = 0  # start frequency
    f2 = fsamp/2 # stop frequency
    Nsamp = buffersize  # number of samples in the chirp
    chirptrain, chirpIn = makeChirpTrain(f1,f2,Nsamp,fsamp,Nchirps)
    # make matrix with 3 columns, each column has signal for each channel: 1, 2, 3
    chirpmatrix1 = np.vstack((chirptrain,np.zeros_like(chirptrain),chirptrain)).T 
    
    # send to soundcard and record
    recordedchirp1 = RMEplayrec(AmpChirp*chirpmatrix1,fsamp,SC=21,buffersize=buffersize)

    chirpmatrix2 = np.vstack((np.zeros_like(chirptrain),chirptrain,chirptrain)).T
    
    time.sleep(0.5)
    recordedchirp2 = RMEplayrec(AmpChirp*chirpmatrix2,fsamp,SC=21,buffersize=buffersize)
    #recordedchirp2 = recordedchirp1

    # process the recorded chirps

    # high pass filter
    print('msize1:',np.shape(chirpmatrix1))
    print('msize2:',np.shape(chirpmatrix2))

    print('size1:',np.shape(recordedchirp1))
    print('size2:',np.shape(recordedchirp2))
    cutoff = 300 # cutoff frequency for high pass filter to filter out low frequency noise
    recordedchirp1 = butter_highpass_filter(recordedchirp1[:,0], cutoff, fsamp, 1)
    recordedchirp2 = butter_highpass_filter(recordedchirp2[:,0], cutoff, fsamp, 1)
    if plotflag:
        fig,ax = plt.subplots()
        ax.plot(recordedchirp1)
        ax.plot(recordedchirp2)
        ax.title("recorded chirptrains")
        ax.set_xlabel("samples")
        ax.set_ylabel("amplitude")
        plt.show()
    

    
    y_stripSClat1 = recordedchirp1[latency_SC:] # remove the SC latency
    y_reshaped1 = np.reshape(y_stripSClat1[:Nchirps*Nsamp],(Nchirps,Nsamp))
    
    y_stripSClat2 = recordedchirp2[latency_SC:] # remove the SC latency
    y_reshaped2 = np.reshape(y_stripSClat2[:Nchirps*Nsamp],(Nchirps,Nsamp))

    # take the mean across some responses:
    Nchskip = 10 # skip first ten chirps

    y_mean1 = np.mean(y_reshaped1.T[:,Nchskip:],axis=1)
    y_mean2 = np.mean(y_reshaped2.T[:,Nchskip:],axis=1)
    if plotflag:
        fig,ax = plt.subplots()
        ax.plot(y_mean1)
        ax.plot(y_mean2)
        ax.title("mean value of the recorded chirps")
        ax.set_xlabel("samples")
        ax.set_ylabel("amplitude")
        plt.show()
    

    Nmean1 = len(y_mean1)  # length of the data
    NmeanUp1 = int(2**np.ceil(2+np.log2(Nmean1)))  # interpolated length of the spectrum
    ChResp1 =  np.fft.rfft(y_mean1,NmeanUp1)/np.fft.rfft(AmpChirp*chirpIn,NmeanUp1)
    fxCh1 = np.arange(NmeanUp1)*fsamp/NmeanUp1
    fxCh1 = fxCh1[:NmeanUp1//2+1]

    Nmean2 = len(y_mean2)  # length of the data
    NmeanUp2 = int(2**np.ceil(2+np.log2(Nmean2)))  # interpolated length of the spectrum
    ChResp2 =  np.fft.rfft(y_mean2,NmeanUp2)/np.fft.rfft(AmpChirp*chirpIn,NmeanUp2)
    fxCh2 = np.arange(NmeanUp2)*fsamp/NmeanUp2
    fxCh2 = fxCh2[:NmeanUp2//2+1]   

    # to smooth out noise, Savitzky-Golay filter is used
    ChRespAbs1 = savgol_filter(np.abs(ChResp1), 20, 2)
    ChRespImag1 = savgol_filter(np.unwrap(np.angle(ChResp1)), 20, 2)

    ChRespAbs2 = savgol_filter(np.abs(ChResp2), 20, 2)
    ChRespImag2 = savgol_filter(np.unwrap(np.angle(ChResp2)), 20, 2)
      
    # change on 8.3.2024 to remove miccal
    #MicC = loadmat('MicCalCurve.mat')  # load calibration curve for the microphone
    #fx = MicC['fx'][0]
    #Hoaemic = MicC['Hoaemic'][0]
    fx = np.arange(100,18e3,5)

    ChRespI1 = np.interp(fx,fxCh1,ChRespAbs1)*np.exp(1j*np.interp(fx,fxCh1,ChRespImag1))
    ChRespI2 = np.interp(fx,fxCh2,ChRespAbs2)*np.exp(1j*np.interp(fx,fxCh2,ChRespImag2))

    #Hinear1 = ChRespI1/(Hoaemic*(10**(MicGain/20))) # convert recorded response to Pascals
    #Hinear2 = ChRespI2/(Hoaemic*(10**(MicGain/20)))
    Hinear1 = ChRespI1/(0.003*10**(MicGain/20))
    Hinear2 = ChRespI2/(0.003*10**(MicGain/20))

    fxinear = fx
    return Hinear1, Hinear2, fxinear, y_mean1, y_mean2



def inst_freq_synch_swept_sine(fstart,fstop,T,fsamp):
    '''
    returns instantaneous frequencies for the synch. swept sine

    '''

    L = T/np.log(fstop/fstart) 
    t = np.arange(0,T,1/fsamp)
    finst = fstart*np.exp(t/L)  # instantaneous frequency
    return finst


def synchronized_swept_sine_spectra_shifted(f1s,L1sw,fsamp,Nsamp,tshift):
    '''
    returns SSS in freq domain
    '''
    # frequency axis
    #f = np.linspace(0,fsamp,Nsamp,endpoint=False)
    f = np.linspace(0,fsamp/2,num=round(Nsamp/2)+1)  # half of the spectrum
    # spectra of the synchronized swept-sine signal [1, Eq.(42)]
    X = 1/2*np.emath.sqrt(L1sw/f)*np.exp(1j*2*np.pi*f*L1sw*(1 - np.log(f/f1s)) - 1j*np.pi/4)*np.exp(-1j*2*np.pi*f*tshift)
    X[0] = np.inf # protect from devision by zero
    return (X,f)

def roexwin(N,n,fs,tc1,tc2):
    # construct window comosed of two halfs of roex windows
    # see Kalluri and Shera J. Acoust. Soc. Am. 109 (2), February 2001 p.622
    #tx = 0:1/fs:(N/2-1)/fs;
    tx = np.arange(0,round(N/2)/fs,1/fs)

    def GetExpFilt(t,tc,N):
        g0=1
        for i in range(1,N):
            g0=np.log(g0+1)
        
        T=np.sqrt(g0)*t/tc
        G=np.exp((T**2))
        for i in range(1,N):
            G=np.exp(G-1)
        
        return 1/G    

    G1 = GetExpFilt(tx,tc1,n)

    G2 = GetExpFilt(tx,tc2,n)

    G = np.concatenate((np.flip(G1), G2))

    return G


def fceDPOAEinwinSSNoise(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tau02,tshift):
    '''
    % function [Hm, fx, hm, tx, hmnoise] = fceDPOAEinwinSSNoise(oaeDS,Npts,f1s,L1sw,rF,fsD,tau01,tau02,tshift)
    %
    % calculates Noise floor with SSS technique
    % 
    % oaeDS - time domain signal from which DPOAE is extracted (response to
    % swept sines in the ear canal)
    % Npts - number of samples in oaeDS
    % fs1 - starting frequency of the synch. swept sine (the initial frequency
    % regardless windowing)
    % L1sw - parameter for the swept sine: L1sw = T/log(f2/f1) (for the overall
    % swept sine)
    % rF - ratio of the f2/f1 frequencies used to evoke DPOAE
    % fsD - sampling frequency
    % tau01 - tau parametMicCalFN = 'CalibMicData.mat'er (cutoff) for first half of the recursive exponential function
    % tau01 - tau parameter (cutoff) for the second half of the recursive exponential function
    % region around the DPOAE impulse response
    % tshift - time shift (for the swept sine response in the windows)

    % calculate spectrum of synchronized swept sine (shifted version in case of
    % time windows which does not start at 0 time)
    '''

    fft_len = int(2**np.ceil(np.log2(len(oaeDS)))) # number of samples for fft 
    S,f = synchronized_swept_sine_spectra_shifted(f1s,L1sw,fsamp,fft_len,tshift)

    # spectra of the output signal
    
    Y = np.fft.rfft(oaeDS,fft_len)/fsamp  # convert the response to frequency domain
    
    # frequency-domain deconvolution
    H = Y/S
    h = np.fft.irfft(H)  # calculated "virtual" impulse response
    
    rF1a = 2 - rF  # 2*f1-f2 component
    rF1b = 3 - 2*rF # for 3*f1-2*f2 component
    rF1 = np.mean((rF1a,rF1b))  # calculate the mean to be between these to Fdp components for noise estimation
    dt = -fsamp*L1sw*np.log(rF1)     # positions of the selected (coef2) IMD component [samples]
    dt_ = round(dt)
    dt_rem = dt - dt_
    hmfftlen = 2**12
    len_IRpul = int((hmfftlen)/2) # length of the impulse response window (adequate for the used time windows for DPOAEs)
    if dt_>0:  # upward sweep
        hm = h[dt_-len_IRpul:dt_+len_IRpul]
    else:  # downward sweep
        hm = h[len(h)+dt_-len_IRpul:len(h)+dt_+len_IRpul]
   
    
    axe_w = np.linspace(0,np.pi,len_IRpul+1,endpoint=False)

    # Non-integer sample delay correction
    Hx = np.fft.rfft(hm) * np.exp(-1j*dt_rem*axe_w)
    hm = np.fft.irfft(Hx)
    
    # apply roex windows to suppress noise and perform component separation
    Nwindow = 10 # degree of roex windows
    w = roexwin(len(hm),Nwindow,fsamp,tau01,tau02)
    
    hm = hm*w  # multiply with roex window
    # add zeros to achieve larger number of points
    hm = np.concatenate((np.zeros(2**11),hm,np.zeros(2**11)))
    Hm = np.fft.rfft(np.fft.fftshift(hm))

    
    hmfftlen = len(hm)

    fxall = np.arange(hmfftlen)*fsamp/hmfftlen # overall freq axis
    fx = fxall[:round(hmfftlen/2)+1]
    return Hm, fx

def fceDPOAEinwinSS(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tau02,tshift):
    '''
    % function [Hm, fx, hm, tx, hmnoise] = fceDPOAEinwinSS(oaeDS,Npts,f1s,L1sw,rF,fsD,tau01,tau02,tshift)
    %
    % calculates DPOAE with SSS technique
    % 
    % oaeDS - time domain signal from which DPOAE is extracted (response to
    % swept sines in the ear canal)
    % Npts - number of samples in oaeDS
    % fs1 - starting frequency of the synch. swept sine (the initial frequency
    % regardless windowing)
    % L1sw - parameter for the swept sine: L1sw = T/log(f2/f1) (for the overall
    % swept sine)
    % rF - ratio of the f2/f1 frequencies used to evoke DPOAE
    % fsD - sampling frequency
    % tau01 - tau parametMicCalFN = 'CalibMicData.mat'er (cutoff) for first half of the recursive exponential function
    % tau01 - tau parameter (cutoff) for the second half of the recursive exponential function
    % region around the DPOAE impulse response
    % tshift - time shift (for the swept sine response in the windows)

    % calculate spectrum of synchronized swept sine (shifted version in case of
    % time windows which does not start at 0 time)
    '''

    fft_len = int(2**np.ceil(np.log2(len(oaeDS)))) # number of samples for fft 
    S,f = synchronized_swept_sine_spectra_shifted(f1s,L1sw,fsamp,fft_len,tshift)

    # spectra of the output signal
    
    Y = np.fft.rfft(oaeDS,fft_len)/fsamp  # convert the response to frequency domain
    
    # frequency-domain deconvolution
    H = Y/S
    h = np.fft.irfft(H)  # calculated "virtual" impulse response
    
    rF1 = 2 - rF
    dt = -fsamp*L1sw*np.log(rF1)     # positions of the selected (coef2) IMD component [samples]
    dt_ = round(dt)
    dt_rem = dt - dt_
    hmfftlen = 2**12
    len_IRpul = int((hmfftlen)/2) # length of the impulse response window (adequate for the used time windows for DPOAEs)
    if dt_>0:  # upward sweep
        hm = h[dt_-len_IRpul:dt_+len_IRpul]
    else:  # downward sweep
        hm = h[len(h)+dt_-len_IRpul:len(h)+dt_+len_IRpul]
   
    
    axe_w = np.linspace(0,np.pi,len_IRpul+1,endpoint=False)

    # Non-integer sample delay correction
    Hx = np.fft.rfft(hm) * np.exp(-1j*dt_rem*axe_w)
    hm = np.fft.irfft(Hx)
    
    #from scipy.io import savemat
    #data = {'hm':hm}
    #savemat('hms015python.mat',data)
    
    # apply roex windows to suppress noise and perform component separation
    Nwindow = 10 # degree of roex windows
    w = roexwin(len(hm),Nwindow,fsamp,tau01,tau02)
    
    hm = hm*w  # multiply with roex window
    # add zeros to achieve larger number of points
    hm = np.concatenate((np.zeros(2**11),hm,np.zeros(2**11)))
    Hm = np.fft.rfft(np.fft.fftshift(hm))

    
    hmfftlen = len(hm)

    fxall = np.arange(hmfftlen)*fsamp/hmfftlen # overall freq axis
    fx = fxall[:round(hmfftlen/2)+1]
    return Hm, fx
 
def ValToPa(MicCalFN,Val,fxV,MicGain):
    # convert spectrum recorded by the sound card into sound pressure in Pa
    # it loads the calibration curve for the microphone in the probe

    MicC = loadmat(MicCalFN)  # load calibration curve for the microphone
    fx = MicC['fx'][0]  # it is assumed that the file has fx variable for freq axis
    Hoaemic = MicC['Hoaemic'][0] # and Hoaemic for the Y axis (complex cal. curve)
    #Spect[0] = Spect[1]

    if isinstance(fxV, np.ndarray):      # for a vector of frequencies
        
        closest_values = [fx[np.argmin(np.abs(fx - val))] for val in fxV]
        matching_indexes = np.where(np.isin(fx, closest_values))[0]
    else:
        differences = np.abs(fx - fxV)

        # Find the index with the minimum difference
        closest_index = np.argmin(differences)

        # Get the value in fx that is closest to fdp
        closest_value = fx[closest_index]
        matching_indexes = np.where(fx==closest_value)[0] # find the index equal to freq value
    

    SpectPa = Val/(Hoaemic[matching_indexes]*(10**(MicGain/20)))
    
    
    return SpectPa


def spectrumToPaFAV(Spect,fxSpect,MicGain):
    # convert spectrum recorded by the sound card into sound pressure in Pa
    
    
    SpectPa = Spect/(0.003*(10**(MicGain/20)))
        
    return SpectPa, fxSpect



def calcDPgramFAV_HS(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,GainMic):
    # function which calculates DP-gram from swept sine response

    rF = f2f1 # frequency ratio f2/f1
    
    Nsamp = len(oaeDS)  # number of samples in the recorded response

    resOct = 3  # resolution 3 - 1/3 oct bands, 2 - 1/2 oct bands
    if f2e>f2b: # upward sweep
        T = np.log2(f2e/f2b)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        fc = np.array([f2b*2**(1/(resOct*2))])
        f2eH = f2e
    else: # downward sweept
        T = np.log2(f2b/f2e)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        fc = np.array([f2e*2**(1/(resOct*2))])
        f2eH = f2b

    f1s = f2b/rF # calculate starting frequency for f1 tone
    f1e = f2e/rF # calculate stop frequnecy for f2 tone
    
    L1sw = T/np.log(f1e/f1s) # overall L1sw parameter for synchronized swept sine

    
    
    while True:
        newfcVal = fc[-1]*2**(1/resOct) # calculate new center frequency for 1/2 oct bands
        if newfcVal < f2eH:
            fc = np.append(fc,newfcVal)
        else:
            break
        
        
    
    fdpc = 2*fc/rF - fc; # convert center freq from f1 to fdp (assumed cdt (low-side))
    if f2e<f2b: # downward sweep, flip the frequencies
        fdpc = fdpc[::-1]
    
    
    tauNL = 0.005*(fdpc/1000)**(-0.8)  # NL window width as a function of fdp
    tauAll = 0.02*(fdpc/1000)**(-0.8) # overall window width (NL + CR components) as a function of fdp

    tau01 = 0.5e-3 # tau for the first half of the roex windows (the part for negative times)

    tshift = 0 # always at time zero is the begining of the window
    
    hmfftlen = 2**13
    Hm = np.zeros((round(hmfftlen/2)+1,len(fdpc)),dtype=complex)
    HmNL = np.zeros((round(hmfftlen/2)+1,len(fdpc)),dtype=complex)
    HmNoise = np.zeros((round(hmfftlen/2)+1,len(fdpc)),dtype=complex)
    HmNoiseNL = np.zeros((round(hmfftlen/2)+1,len(fdpc)),dtype=complex)
    
        
    for k in range(len(fdpc)):  # estimate for individual roex windows
        Hm[:,k], fxfdp = fceDPOAEinwinSS(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tauAll[k],tshift)
        #print(np.isnan(Hm))
        HmNL[:,k], fxfdpNL = fceDPOAEinwinSS(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tauNL[k],tshift)
        HmNoise[:,k], fxfdp = fceDPOAEinwinSSNoise(nfloorDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tauAll[k],tshift)
        HmNoiseNL[:,k], fxfdpNL = fceDPOAEinwinSSNoise(nfloorDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tauNL[k],tshift)
        # make a matrix with NaNs in the respective parts of the responses, i.e. for specific rows in Hm
        # the matrix will then be used to combine Hm and HmNL such that we
        # use concerete roex windows for concrete frequency ranges. This
        # method allows for variable roex window width, but avoid
        # transients at the band edges if the response was processed in
        # time windows of a specific size
        
        fdpcL = fdpc[k]/(2**(0.5/resOct))  # low freq of band
        fdpcH = fdpc[k]*(2**(0.5/resOct))  # high freq of band
        
        if k==0:  # first time window
            matrixNaN = np.ones((len(fxfdp),len(fdpc)))
            idx1 = np.argwhere(fxfdp>=fdpcL)[0][0]
            #idx2 = find(fxfdp>=fdpc(k+1),1,'first');
            idx2 = np.argwhere(fxfdp>=fdpcH)[0][0]
        elif k==len(fdpc)-1: # last time window
            #idx1 = find(fxfdp>=fdpc(k-1),1,'first');
            idx1 = np.argwhere(fxfdp>=fdpcL)[0][0]
            #idx2 = find(fxfdp>=fdpc(k+1),1,'first');
            idx2 = np.argwhere(fxfdp>=fdpcH)[0][0]
        else: # the rest
            #idx1 = find(fxfdp>=fdpc(k-1),1,'first');
            idx1 = np.argwhere(fxfdp>=fdpcL)[0][0]
            #idx2 = find(fxfdp>=fdpc(k+1),1,'first');
            idx2 = np.argwhere(fxfdp>=fdpcH)[0][0]
            #idx2 = find(fxfdp>=fdpc(k+1),1,'first');
        
        
        matrixNaN[:idx1,k] = np.nan
        matrixNaN[idx2:,k] = np.nan


    dpgramAll = np.nanmean((Hm*matrixNaN).T,0)  # NL component
    dpgramNL = np.nanmean((HmNL*matrixNaN).T,0)  # NL component
    noisefAll = np.nanmean((HmNoise*matrixNaN).T,0)  # NL component
    noisefNL = np.nanmean((HmNoiseNL*matrixNaN).T,0)  # NL component
    
  
    MicGain = GainMic
        
    DPOAEcalib,fxI = spectrumToPaFAV(dpgramAll,fxfdp,MicGain)
    DPOAEcalibNL,fxI = spectrumToPaFAV(dpgramNL,fxfdp,MicGain)
    DPOAEcalibCR = DPOAEcalib - DPOAEcalibNL # rest, LL component
    NFLOORcalib,fxI = spectrumToPaFAV(noisefAll,fxfdp,MicGain)
    NFLOORcalibNL,fxI = spectrumToPaFAV(noisefNL,fxfdp,MicGain)
    
    return DPOAEcalib, DPOAEcalibNL, DPOAEcalibCR, NFLOORcalib, NFLOORcalibNL, fxI



def getSClat(fsamp,buffersize):

    # make short chirp
    f1 = 1000
    f2 = 1200
    Nsamp = 2048*20  # cca 500ms
    #fsamp = 44100 # samplinMatN[:,i]g rate must be the same as in jack setup

    #buffersize = 2048  # buffersize for sounddevice stream


    def makeChirp(f1,f2,Nsamp,fsamp):
        T = Nsamp/fsamp  # duration of the chirp
        
        tx = np.arange(0,Nsamp/fsamp,1/fsamp)  # time axis

        A = 1
        T = Nsamp/fsamp
        
        y = np.zeros_like(tx)
        print(len(tx))
        for k in range(len(tx)):
            y[k] = A*np.sin(2*np.pi*(f1+(f2-f1)/(2*(T))*tx[k])*tx[k])

        return y

    y = makeChirp(f1,f2,Nsamp,fsamp)

    # fade in fade out of the chirp
    rampdur = 10e-3 # duration of onset offset ramp

    rampts = np.round(rampdur * fsamp)

    step = np.pi/(rampts-1)
    x= np.arange(0,np.pi+step,step)
    offramp = (1+np.cos(x))/2
    onramp = (1+np.cos(np.flip(x)))/2
    o=np.ones(len(y)-2*len(x))
    wholeramp = np.concatenate((onramp, o, offramp)) # envelope of the entire ramp

    y = y*wholeramp

    # add some zeros
    generated_signal = np.concatenate((np.zeros(4800), 0.1*y, np.zeros(4800)))  # add some zeros to begining

    '''
    fig,ax = plt.subplots()
    ax.plot(generated_signal)
    plt.show()
    '''
    # now send the input to the sound card and record response



    import sys
    import os

    current = os.path.dirname(os.path.realpath(__file__)) # find path to the current file
    UMdir = os.path.dirname(current)+'/UserModules'  # set path to the UserModules folder
    sys.path.append(UMdir) # add the path to the module into sys.path
    from pyRMEsd import RMEplayrec  # import needed functions from pyRMEsd module

    #SC = choose_soundcard() # choose sound card (which device to use)
    SC = 21
    print(f'Chosen device is {SC}')
    chan_in = 3
    sig_in = np.tile(generated_signal,(chan_in,1)).T

    recorded_data = RMEplayrec(sig_in,fsamp,SC=SC,buffersize=buffersize)
    #from pyDPOAEmodule import sendDPOAEstimulustoRMEsoundcard

    '''
    #recorded_data = sendDPOAEstimulustoRMEsoundcard(generated_signal,3,3,fsamp)
    #print(np.shape(output))
    fig,ax = plt.subplots()
    ax.plot(recorded_data)
    ax.plot(generated_signal)
    #ax.set_ylim((-0.001,0.001))
    plt.show()
    '''

    # estimate delay

    

    def lag_finder(y1, y2, sr):
        n = len(y1)

        corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

        delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
        delay = delay_arr[np.argmax(corr)]
        print('y2 is ' + str(delay) + ' seconds behind y1, which is ' + str(np.argmax(corr)-round(n/2)) + ' samples')
        return np.argmax(corr)-round(n/2)
        '''
        plt.figure()
        plt.plot(delay_arr, corr)
        plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coeff')
        plt.show()
        '''



    latSC = lag_finder(generated_signal,recorded_data[:len(generated_signal),2],fsamp)


    return latSC