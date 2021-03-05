import numpy as np
import matplotlib.pyplot as plt
#__name__ = "__main__ "

def StripesUnit8(size= (0,0) ,strips= (0,0) ,numberOfStrips=0 , stripsValueArray= [0,0] ):
    matInit = np.dot(stripsValueArray[0],np.ones(size))
    for i in range(0,numberOfStrips):
        matInit[0:strips[0],i*strips[1]:strips[1]+i*strips[1]] = stripsValueArray[i]
    return matInit
    
def ShowGrayImg(img):
    img = plt.imshow(img,cmap='gray', vmin=0, vmax=255)
    img = plt.gca()
    img.get_xaxis().set_visible(False)
    img.get_yaxis().set_visible(False)
    plt.show()
def AlocateMatrix(index,matrix):
    
    if index == 0 :
        return np.uint8(np.abs(matrix))
    elif index == 1:
        return np.uint8(matrix)
    elif index == 2:
        return matrix
    elif index == 3:
        return np.uint8(matrix - np.amin(matrix))
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
    fig.show()

if __name__ == "__main__ ":
    StripesUnit8((200,300),(200,50),6,[10 ,50 ,100, 150, 200 ,250])