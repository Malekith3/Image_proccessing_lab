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
"""
The function receives a signal and returns it quantized to the given number of bits
  
signal   -- numpy 1d array pref numpy,
            dtype uint8 or float64 in range [0,1]
numBits -- integer between 1 and 8 inclusive
"""
def Quantize1D(signal, numBits):


  if (type(signal[0]) != np.uint8): # float64
    tmpSig = (signal * 255).astype('uint8')
  else:
    tmpSig = signal
  # uint8 is 8 bit, so shifting right by 8 - numBits will quantize
  quantized = np.right_shift(tmpSig, 8 - numBits)
  # shift left to return to the same values
  quantized = np.left_shift(quantized, 8 - numBits)
  # add half of the step size so that the output values are in the middle of the quantized level
  quantized += np.floor(pow(2, (8 - numBits) - 1)).astype('uint8')
  if (type(signal[0]) != np.uint8): # float
    return quantized.astype('float64') / 255 # scale back to [0,1] float
  else:
    return quantized