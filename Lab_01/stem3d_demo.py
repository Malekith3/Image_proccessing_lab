
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from Functions import *
import math as math
from scipy import signal
import cv2 as cv

x= np.array([[2,3,1],[1,4,0]])
h = np.zeros((2*np.size(x,0)+1,3*np.size(x,1)+1),dtype='uint8')
h[0::2,0::3] = 1
y = signal.convolve2d(x,h)
xs = np.array([[i]*12 for i in range(1,7)])
ys= np.array([[i for i in range(1,13)]]*6)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d')) 
ax.plot(xs[0,0],ys[0,0],y[0,0])
plt.show()