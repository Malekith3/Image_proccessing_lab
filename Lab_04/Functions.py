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