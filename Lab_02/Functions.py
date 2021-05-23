#Functions defenition
#__name__ = "__main__"

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import itertools

axis_font = {'fontname':'Arial', 'size':'16'}

def Transform(data,addition=50,multiplication=0.4):
  return (data * multiplication + addition).astype(type(data[0]))

def InverseTransform(data,addition=50,multiplication=0.4):
    return ((data - addition) / multiplication).astype(type(data[0]))

def TransformUsingLookup(data,addition=0,multiplication=1):
  if not hasattr(TransformUsingLookup, "lookUpTable"): 
    TransformUsingLookup.lookUpTable = Transform(np.arange(256),addition,multiplication)
  return np.take(TransformUsingLookup.lookUpTable, data)

def InverseTransformUsingLookup(data,addition=0,multiplication=1):
  if not hasattr(InverseTransformUsingLookup, "lookUpTable"): 
    InverseTransformUsingLookup.lookUpTable = InverseTransform(np.arange(256),addition,multiplication)
  return np.take(InverseTransformUsingLookup.lookUpTable, data)

#-----------------------------------------1.3.5 functions------------------------------------------------------------
def TransformUsingLookupFloat(data,dup=0.4,add=50.0/255.0):
  if not hasattr(TransformUsingLookupFloat, "lookupTable"): 
    TransformUsingLookupFloat.lookupTable = Transform(np.arange(256, dtype = 'float64') / 255)
  return np.take(TransformUsingLookupFloat.lookupTable, np.uint8(data*255))

def InverseTransformUsingLookupFloat(data,dup=0.4,add=50.0/255.0):
  if not hasattr(InverseTransformUsingLookupFloat, "lookupTable"): 
    InverseTransformUsingLookupFloat.lookupTable = InverseTransform(np.arange(256, dtype = 'float64') / 255)
  return np.take(InverseTransformUsingLookupFloat.lookupTable,  np.uint8(data*255))
#---------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------
def Histogram(image,typeOfPlot = "histogram"):
  checkArray = np.array(image)
  histArray = [None] * 256


  # Histogram 

  for i in range(len(histArray)):
    histArray[i] = np.count_nonzero(checkArray == i)

  # Normalized Histogram 

  histmax = np.amax(histArray)
  histMin = np.amin(histArray)
  histNormArray = np.array([(i - histMin) / (histmax - histMin) for i in histArray])
    
  # Cumulative Histogram 

  cumulativeHist = np.zeros(256)
  cumulativeHist[0] = histNormArray[0]
  for i in range(1,256):
    cumulativeHist[i] = cumulativeHist[i-1] + histNormArray[i]

  cumulativeMax = np.amax(cumulativeHist)
  cumulativeMin = np.amin(cumulativeHist)

  cumulativeHist = [(i - cumulativeMin) / (cumulativeMax-cumulativeMin) for i in cumulativeHist]
  imageHistArray = [np.arange(0,256),np.array(histArray),np.array(histNormArray),np.array(cumulativeHist)]

  listOfPlots = {"Image":image,"Histogram":histArray,"Normalized":histNormArray,"Cumulative":cumulativeHist}

  PrintHistogram(image,listOfPlots,typeOfplot=typeOfPlot)

  # return array which is --> [Image values, histogram values, normalized histogram values, cumulative histogram values]
  if typeOfPlot.lower() == "array": 
    return imageHistArray
 
#--------------------------------------------Help Functions of Histogram--------------------------------------------------------------

def PrintHistogram(img,listOfPlots = {} , typeOfplot="histogram"):
  titles = ["","Histogram","Normalized","Cumulative"]
  xlable = "Pixel Value"
  ylables = ["","Quantity","Norm Quantity","Norm Cumulative Quantity"]

  if typeOfplot.lower() == "histogram":

    plt.plot(listOfPlots["Histogram"])
    plt.title(titles[0],**axis_font)
    plt.xlabel(xlable,**axis_font)
    plt.ylabel(ylables[0],**axis_font)

  elif typeOfplot.lower() == "normalized":

    plt.plot(listOfPlots["Normalized"])
    plt.title(titles[1],**axis_font)
    plt.xlabel(xlable,**axis_font)
    plt.ylabel(ylables[1],**axis_font)

  elif typeOfplot.lower() == "cumulative":

    plt.plot(listOfPlots["Cumulative"])
    plt.title(titles[2],**axis_font)
    plt.xlabel(xlable,**axis_font)
    plt.ylabel(ylables[2],**axis_font)

  elif typeOfplot.lower() == "all":
      figure, ax = plt.subplots(4,1,figsize=[10,10])
      for (title,a,ylable,flag) in itertools.zip_longest(titles,ax,ylables,range(0,1)):
        if flag == 0 :
          a.imshow(img,cmap='gray')
          a.get_xaxis().set_visible(False)
          a.get_yaxis().set_visible(False)
        else:
          a.plot(listOfPlots[title])
          a.set_title(title,**axis_font)
          a.set_xlabel(xlable,**axis_font)
          a.set_ylabel(ylable,**axis_font)
#-------------------------------------------------------------------------------------------------------------------------------------
"""
Function of histogram stretching
  (0,0) - (x1, y1)
  (x1, y1) - (x2, y2)
  (x2, y2) - (255, 255)
  Input :
  grayscale image uint or float64 in range [0.0,1.0]
  x1,y1,x2,y2 - uint8
  Assuming:
  x1 > 0
  x1 < x2 < 255
  y1 < y2
"""       
def HistogramStretch(img, x1 = 1, y1 = 1, x2 = 254, y2 = 254,reload=False):
  #converting imag in float to uint8 for LUT
  is_float = (np.dtype(img[0][0]) == 'float64')
  if (is_float):
    img = (img * 255).astype('uint8')
  

  #setting up LUT
  if not hasattr(HistogramStretch, "lookupTable") or reload==True:
    ramp1 = float(y1/x1)
    ramp2 = float ((y2-y1)/(x2-x1))
    ramp3 = float((255-y2)/(255-x2))
    temp = []
    for i in range(0,256):
      if i < x1:
        temp.append((ramp1 * i))
      elif i <= x2:
        temp.append((y1 + ramp2 * (i - x1)))
      else:
        temp.append((y2 + ramp3 * (i - x2)))
    HistogramStretch.lookupTable = np.array(temp,dtype='uint8')

  output = np.take(HistogramStretch.lookupTable, img)

  if (is_float):
    return (output / 255.0)
  else:
    return output

#-------------------------------------------------------------------------------------------------------------------------------------
def HistogramFullStreach(img):
  is_float = (np.dtype(img[0][0]) == 'float64')
  if (is_float):
    img = (img * 255).astype('uint8')

  temp = []
  ramp = 1/(np.amax(img) - np.amin(img))
  for i in range(256):
    temp.append(255*((i-np.amin(img))*ramp))
  temp = np.array(temp,dtype='uint8')
  output = np.take(temp, img)

  if (is_float):
    return (output / 255.0)
  else:
    return output
"""
function to calculate MSE 
inputs: image - original image
        noise - Image with noise 
output: MSE - matrix that represents MSE
"""
def MSEOfTwoImages(image,noise):
    tmpSum = np.sum((image - noise)**2,dtype='float64')
    return round(tmpSum/(np.size(image)),2)
#-------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
  img_gray = cv.imread("Pictures\\tire.tif", cv.IMREAD_GRAYSCALE)
plt.plot(HistogramFullStreach(np.expand_dims(np.arange(256, dtype = 'uint8'), axis = 0)))
plt.show()