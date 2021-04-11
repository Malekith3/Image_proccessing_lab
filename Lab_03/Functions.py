import numpy as np
import matplotlib.pyplot as plt
import itertools
#-------------------------------------------------------------------------------------------------------------
"""
Input:parameter to sample cosine function
Output: Sampled cosine or sine
"""
def SampledCosine(timeLenght=1,freqSample = 200 , cosineFreq = 10 ,amplitude =7,returnSinus = False ,returnLinspace=True):
    t = np.dot(1/freqSample,[i for i in range(freqSample)])
    if(returnLinspace):
        if(not returnSinus):
            return np.array((amplitude*np.cos(cosineFreq*t*2*np.pi))),t
        else :
            return np.array((amplitude*np.sin(cosineFreq*t*2*np.pi))),t
    else:
        if(not returnSinus):
            return np.array((amplitude*np.cos(cosineFreq*t*2*np.pi)))
        else :
            return np.array((amplitude*np.sin(cosineFreq*t*2*np.pi)))
#-------------------------------------------------------------------------------------------------------------
def PlotingSignalWithFFT(plots,xAxis,titles,xtitle,ytitle):
    figure ,axis = plt.subplots(2,3,figsize=[15,8])
    for (ax,plot,title,xl,yl,xAx) in itertools.zip_longest(axis,plots,titles,xtitle,ytitle,xAxis):
        for (a,pl,title,xlabel,ylabel,x) in itertools.zip_longest(ax,plot,title,xl,yl,xAx):
            a.plot(x,np.real(pl))
            a.set_title(title)
            a.set_xlabel(xlabel)
            a.set_ylabel(ylabel)
    plt.tight_layout()
#-------------------------------------------------------------------------------------------------------------
"""
Input:signal and same signal with noise
Output: MSE , SNR , PSNR
"""
def ImageDataAnalysis(signal,noisedSignal):
    mse = (np.square(signal - noisedSignal)).mean()
    error = (signal-noisedSignal)
    sigMean = (np.square(signal)).mean()
    noiseMean = (np.square(error)).mean()

    snr = 10*np.log10(sigMean/noiseMean)
    psnr = 20*np.log10(255) - mse # 255 is the maximum grey value

    return np.real(mse), round(np.real(snr),3), round(np.real(psnr),3)
    