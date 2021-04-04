import numpy as np
import matplotlib.pyplot as plt


def StripesUnit8(size= (0,0) ,strips= (0,0) ,numberOfStrips=0 , stripsValueArray= [0,0] ):
    matInit = stripsValueArray[0] * np.ones(size)
    for i in range(0,numberOfStrips):
        matInit[0:strips[0],i*strips[1]:strips[1]+i*strips[1]] = stripsValueArray[i]
    return matInit
#-------------------------------------------------------------------    
def ShowGrayImg(img,title=""):
    img = plt.imshow(img,cmap='gray', vmin=0, vmax=255)
    img = plt.gca()
    img.get_xaxis().set_visible(False)
    img.get_yaxis().set_visible(False)
    img.set_title(title)
    plt.show()
#-------------------------------------------------------------------
def AlocateMatrix(index,matrix):
    
    if index == 0 :
        return np.uint8(np.abs(matrix))
    elif index == 1:
        return np.uint8(np.clip(matrix,0,255))
    elif index == 2:
        return matrix
    elif index == 3:
        return np.uint8(matrix - np.amin(matrix))
#-------------------------------------------------------------------       
def Normalization(matrix):
    ret = matrix.astype('float64')
    minimum = np.min(ret)
    maximum = np.max(ret)
    ret = (ret - np.ones(ret.shape) * minimum) / ((maximum - minimum) * 1.001)
    return ret
#-------------------------------------------------------------------
def PlotingDiscreteConvolution(x,h,y,titile):
    axis_font = {'fontname':'Arial', 'size':'18'}
    fig , subplotlist = plt.subplots(3,1,figsize=[10,5],sharey=True,sharex=True)
    plt.tight_layout()
    subplotlist[0].set_title(titile)
    subplotlist[2].set_xlabel("n",**axis_font)
    subplotlist[0].set_ylabel("x[n]",**axis_font)
    subplotlist[1].set_ylabel("h[n]",**axis_font)
    subplotlist[2].set_ylabel("y[n]",**axis_font)
    for subplot in subplotlist:
        subplot.set_ylim((0,np.amax(y)+2))
    #ploting stuff
    subplotlist[0].stem([i for i in range(1,np.size(x)+1)],x)
    subplotlist[1].stem([i for i in range(1,np.size(h)+1)],h)
    subplotlist[2].stem([i for i in range(1,np.size(y)+1)],y)
#-------------------------------------------------------------------
def SpatialSinus(A,fx,fy,fs,Nx,Ny,offset):
    sine = np.zeros((Nx-1,Ny-1),dtype='uint8')
    fx_normalized = fx/fs
    fy_normalized = fy/fs
    for i in range(0,Ny-1):
        for j in range(0,Nx-1):
            sine[i,j] = np.round(offset + A*np.sin(2*np.pi*fy_normalized*i +2*np.pi*fx_normalized*j))
    return sine
#-------------------------------------------------------------------
def AddImages(listOfImages = [""]):
    sumOfImages = np.zeros(np.shape(listOfImages[0]))
    for image in listOfImages:
        sumOfImages += image.astype("float64")
    return np.floor(sumOfImages/len(listOfImages)).astype('uint8')

#-------------------------------------------------------------------
"""
function to calculate MSE 
inputs: image - original image
        noise - Image with noise 
output: MSE - matrix that represents MSE
"""
def MSEOfTwoImages(image,noise):
    tmpSum = np.sum((image - noise)**2,dtype='float64')
    return round(tmpSum/(np.size(image)),2)
#-------------------------------------------------------------------
"""
function to calculate SNR 
inputs: image - original image
        noise - Image with noise 
output: SNR - matrix that represents SNR
"""
def SNROfTwoImages(image,noise):
    Es = np.sum((image - np.average(image))**2  , dtype='float64')/np.size(image)
    En = np.sum((noise - np.average(noise))**2  , dtype='float64')/np.size(noise)
    return round(10*np.log10(Es/En),2)
#-------------------------------------------------------------------

"""
function to calculate PSNR 
inputs: image - original image
        noise - Image with noise  
output: PSNR - matrix that represents PSNR
"""
def PSNROfTwoImages(image,noise):
    MSE = MSEOfTwoImages(image,noise)
    return round(10*np.log10(np.amax(image)**2/MSE),2)

#-------------------------------------------------------------------