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
Quentize given single .
  
signal   -- numpy 1d numpy array,
            dtype uint8 or float64 in range [0,1]
numBits -- integer between 1 and 8 inclusive
"""
def Quantize1D(signal, numBits):


  if (type(signal[0]) != np.uint8):
    tmpSig = (signal * 255).astype('uint8')
  else:
    tmpSig = signal

  quantized = np.right_shift(tmpSig, 8 - numBits)
 
  quantized = np.left_shift(quantized, 8 - numBits)
 
  quantized += np.floor(pow(2, (8 - numBits) - 1)).astype('uint8')
  if (type(signal[0]) != np.uint8): 
    return quantized.astype('float64') / 255 
  else:
    return quantized
#-------------------------------------------------------------------------------------------------------------
def stripes_image(width = 256, height = 256, stripeValues = [0, 100, 200]):
  """"
  Create image with a strips width and height decide size of it and strip values is width of strips 

  width, height -- pixels
  stripeValues -- 1D array with a strip width values 
  """

  stripeWidth = int(width / len(stripeValues))
  x = np.ones([height, stripeWidth], dtype = np.uint8) * stripeValues[0] 
  for i in range(1, len(stripeValues)):
    new_stripe = np.ones([height, stripeWidth], dtype = np.uint8) * stripeValues[i]
    x = np.concatenate((x, new_stripe), axis = 1)

  return x

#-------------------------------------------------------------------------------------------------------------

def Quantize(im, levels, qtype='uniform', maxCount=255, displayLevels=None):

    levels = levels + 1
    returnImage = np.copy(im)
    dtype = im.dtype

    if (displayLevels == None):
        displayCount = levels
    elif displayLevels > 0:
        displayCount = displayLevels-1
    else:
        print("displayLevels is an invalid value")
        return returnImage
        
    if ((levels > 0) and (levels < maxCount)):
        levels = levels - 1
    else:
        print("levels needs to be a positive value, and smaller than the maxCount")
        return returnImage

    if (qtype == 'uniform'):
        
        returnImage = np.floor((im/((maxCount+1)/float(levels))))*(displayCount/levels)
    elif (qtype == 'max_lloyed'):
        histCounts, histBins = np.histogram(im, bins = 256, range = [0.0, 255.0]) 
        histBins = histBins[:histBins.shape[0] - 1] 
        histCountsMulBins = histCounts * histBins 
        levels = levels + 1
        r = np.linspace(0, 255, levels) 
        f = 0.5 * (r[:levels - 1] + r[1:]) 
        maxIterations = 10000 
        for iteration in range(0, maxIterations):
          previous_f = np.copy(f)
          for index in range(0, levels - 1): 
            lowR = r[index]
            highR = r[index + 1]
            
            integralIndexes = np.logical_and(histBins >= lowR, histBins <= highR) 
            
            nominator = np.sum(histCountsMulBins[integralIndexes].astype('float64'))
            denominator = np.sum(histCounts[integralIndexes])
            if (denominator == 0): 
              if (index == 0):
                f[index] = 1
              else:
                f[index] = f[index - 1] + 1
            else:
              f[index] = nominator / denominator

          r[1: levels - 1] = 0.5 * (f[:f.shape[0] - 1] + f[1:]) 

          if (np.sum(np.abs(f - previous_f)) < 0.00001): 
            print("stop condition reached after %d iterations. delta f = " % iteration, end = "")
            print(f - previous_f)
            break

        f = np.round(f)
        for index in range(0, levels - 1): 
          lowR = r[index]
          highR = r[index + 1]
          pixel_indexes = np.logical_and(returnImage >= lowR, returnImage <= highR) 
          returnImage[pixel_indexes] = f[index]
    return np.array(returnImage, dtype)
#-------------------------------------------------------------------------------------------------------------
def MSE_SNR_PSNR(SignalA, SignalB):
    """Calculate MSE , SNR ,PSNR for Image . Returning Valuse is in that order"""
    mse = (np.square(SignalA - SignalB)).mean()
    error_ = (SignalA - SignalB)
    sig_mean = (np.square(SignalA)).mean()
    noise_mean = (np.square(error_)).mean()

    snr = 10*np.log10(sig_mean/noise_mean)
    psnr = 20*np.log10(255/np.sqrt(mse))

    return mse, snr, psnr

#-------------------------------------------------------------------------------------------------------------
def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img