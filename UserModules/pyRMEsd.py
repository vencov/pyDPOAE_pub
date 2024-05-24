# module with functions comunicating with sound card
import sounddevice as sd
import numpy as np
import sys

def show_soundcards():
    print('List of available devices:')
    list_devices = sd.query_devices()  # get list of devices
    print(list_devices)  # print a list of sound devices

    return list_devices

def choose_soundcard():

    list_dev = show_soundcards()
    while True:
        CSC = input('Choose the sound card: ')

        try:
            SC = int(CSC)
            if not (len(list_dev)<SC or SC<0):
                break
        except:
            print('You must write a number of the devices from the list!')

    return SC


def RMEplayrec(dataout,fsamp,*,SC=21,buffersize=2048):
    '''
    RMEplayrec(dataout,fsamp,SC=21,buffersize=2048)
    simultanouse playback and record
    1D input data is sent into first 3 output channels of the sound card and signal is
    recorded from the first 3 input channels of the sound card
    3rd channels are wired connected (sound card latency can be estimated from them)
    dataout - 1D vector with data
    fsamp - sampling frequency
    SC = 21 - device number
    blocksize - buffersize
    '''
    global recorded_data, generated_signal, blocksize,idxPointer
    chan_in = 3  # number of channels for input and output
    output = np.empty((1,chan_in))
    Nsamp = len(dataout)  # number of samples in the signal which is sent to the sound card
    #generated_signal = np.tile(dataout,(chan_in,1)).T
    generated_signal = dataout
    recorded_data = np.zeros((0,chan_in))
    idxPointer = 0
    blocksize = buffersize
    def callback(indata,outdata, frames, time, status):
        global idxPointer, blocksize, generated_signal,recorded_data
        '''
        indata - data recorded by the sound card
        outdata - data sent to the sound card
        '''
        
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort
        
        if status.input_underflow:
            print('Input underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort
        

        recorded_data = np.concatenate((recorded_data, indata), axis=0)
        dataout = generated_signal[idxPointer:idxPointer+blocksize,:]
        if np.shape(dataout)[0] < blocksize:
            outdata[:np.shape(dataout)[0],:] = dataout
            outdata[np.shape(dataout)[0]:,:].fill(0)
            #raise sd.CallbackStop
        else:
            outdata[:] = dataout
        idxPointer+=blocksize
        #print(idxPointer)


    stream = sd.Stream(device=(SC,SC),samplerate=fsamp, blocksize=blocksize,
        channels=chan_in,callback=callback, latency='low')

    import time
    with stream:
        time.sleep((Nsamp+40e3)*1/fsamp)  # with almost save time delay for 2048 buffers

    return recorded_data