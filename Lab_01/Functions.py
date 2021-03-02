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


if __name__ == "__main__ ":
    StripesUnit8((200,300),(200,50),6,[10 ,50 ,100, 150, 200 ,250])