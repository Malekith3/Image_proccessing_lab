
import numpy as np
import matplotlib.pyplot as plt
from Functions import *
import math as math
import time

background = 220
b_object = 230
moving = 2
square_img = np.ones((400,400)) * background
fig1 , histograma = plt.subplots(2,figsize=[8,8])
for i in range (0,10):
    square_img[150:250,150:250] = b_object-i*moving
    histograma[0].imshow(square_img,cmap='gray', vmin=0, vmax=255)
    histograma[1].hist(square_img.ravel(), bins = [i for i in range(0,255)], facecolor='blue', alpha=0.5,stacked=True)
    Cw = ((b_object-i*moving)-background)/background
    plt.title(f"cw is {Cw}")
    fig1.show()
    