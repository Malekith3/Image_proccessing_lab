import numpy as np
import matplotlib.pyplot as plt
import itertools
from  scipy.signal import convolve2d

axis_font = {'fontname':'Arial', 'size':'16'}
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
#-------------------------------------------------------------------------------------------------------------    
def convolve2dFilterAndPlot(img, filter, conv_mode = 'same'):
  """Perform 2D convolution and plot the results and FFT of result
  
  conv_mode:
  full - The output is the full discrete linear convolution of the inputs. (Default)
  valid - The output consists only of those elements that do not rely on the zero-padding
  same - The output is the same size as in1, centered with respect to the ‘full’ output.
  """

  filtered = convolve2d(img, filter, mode = conv_mode)
  figure, axis = plt.subplots(1,3,figsize=[20,6])
  titles = [f"Image, shape = {img.shape}",'log10(1+abs(fft(filtered)))',f"Filtered, shape = {filtered.shape} ({conv_mode})"]
  plots = [img,np.fft.fftshift(np.log10(abs(np.fft.fft2(filtered)) + 1)),filtered]
  for (ax ,title,plot) in itertools.zip_longest(axis,titles,plots):
      ax.imshow(plot, cmap = 'gray', vmin = 0, vmax = np.max(plot))
      ax.set_title(title,**axis_font)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  figure.tight_layout()
  plt.show()

