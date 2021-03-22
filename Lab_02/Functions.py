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
    TransformUsingLookupFloat.lookupTable = transform(np.arange(256, dtype = 'float64') / 255)
  return np.take(TransformUsingLookupFloat.lookupTable, np.uint8(data*255))

def InverseTransformUsingLookupFloat(data,dup=0.4,add=50.0/255.0):
  if not hasattr(InverseTransformUsingLookupFloat, "lookupTable"): 
    InverseTransformUsingLookupFloat.lookupTable = InverseTransform(np.arange(256, dtype = 'float64') / 255)
  return np.take(inverse_transform_using_lookup_float.lookupTable,  np.uint8(data*255))
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
if __name__ == '__main__':
  img_gray = cv.imread("Pictures\\tire.tif", cv.IMREAD_GRAYSCALE)
  Histogram(img_gray,"all")
  plt.show()
  