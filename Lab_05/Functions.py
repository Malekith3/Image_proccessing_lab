import numpy as np
import  cv2 as cv






def square_in_square_image(N = 400, objectStart = 150, objectEnd = 250, B_Background = 0, B_Object = 255):
  """Create Image with solid background and in a center is square with chosen background intensity
     Image is garyscale
  """

  image = np.ones([N, N], dtype = np.uint8) * B_Background # create a new image with background color
  image[objectStart:objectEnd + 1, objectStart:objectEnd + 1] = B_Object
  return image

#-------------------------------------------------------------------------------------------------------------

def NestedSquares(N = 200, Thickness = 20, B_Background = 0 , B_Object = 255, Squares = 5):

  image = np.ones([N,N], dtype = np.uint8) * B_Background
  color = B_Object

  for i in range(Squares-1):
    objectStart = Thickness * (i + 1)
    objectEnd = N - (Thickness * (i + 1))
    image[objectStart:objectEnd + 1, objectStart:objectEnd + 1] = color
    
    if color == B_Object:
      color = B_Background
    else:
      color = B_Object

  return image

  #-------------------------------------------------------------------------------------------------------------
