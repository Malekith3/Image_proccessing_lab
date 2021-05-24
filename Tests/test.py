import numpy as np 

def nested_squares(N = 200, Thickness = 20, B_Background = 0 , B_Object = 255, Squares = 5):

  x = np.ones([N,N], dtype = np.uint8) * B_Background
  color = B_Object

  for i in range(Squares-1):
    object_start = Thickness * (i + 1)
    object_end = N - (Thickness * (i + 1))
    x[object_start:object_end + 1, object_start:object_end + 1] = color
    
    if color == B_Object:
      color = B_Background
    else:
      color = B_Object

  return x



def rle_encode_save_image(binary_image, delimiter = ',', file_name = 'rle_encoded_binary_image.csv', path_to_file = 'content'):
  """
  Encode binary image as file with given delimiter between values.
  binary_image -- numpy ndarray of values 255 and zeros.
  file_name -- must contain the file type, e.g. .csv

  The function returns a number - average bits per pixel. Not including delimiters.
  """
  
  # pre processing - encode the image into a 1D ndarray of lengths, including height and width  
  flattened_image = (binary_image / 255).astype('uint8').flatten() # flatten and turn 255 to ones

  arr_of_lengths = np.empty([0], dtype = 'uint16') # empty ndarray of size 0
  value_to_encode = 0 # initial value - black
  counter = 10000000 # maximum iterations, to avoid infinite while loop
  # print(flattened_image.shape) # num pixels
  while (counter > 0):
    if ((1 - value_to_encode) in flattened_image):
      length = np.argmax(flattened_image == (1 - value_to_encode)) # the first occurence of the value (1 - value_to_encode)
      arr_of_lengths = np.append(arr_of_lengths, length)
      flattened_image = flattened_image[length:] # throw away the written values
    else: # theres only a single value left in flattened_image
      arr_of_lengths = np.append(arr_of_lengths, flattened_image.shape[0])
      break
    counter -= 1
    value_to_encode = 1 - value_to_encode # invert from 1 to 0 and vice versa

  #print(arr_of_lengths)
  #print(np.sum(arr_of_lengths)) # num of pixels
  # since we are using 16 bit and we do not expect to go over 2^16, there is no special attention to long numbers.

  # generate header
  #bits_per_number = 16 # uint16
  encoding_type = 10 # binary RLE
  width = binary_image.shape[1]
  height = binary_image.shape[0]
  header_length_in_numbers = 4 # encoding_type, width, height, header_length_in_numbers

  # save header and arr_of_lengths to a file
  with open(path_to_file + '\\' + file_name, 'w') as f: # 'w' for overwrite
    # header
    f.write('{0:d}'.format(header_length_in_numbers) + delimiter)
    f.write('{0:d}'.format(encoding_type) + delimiter)
    f.write('{0:d}'.format(width) + delimiter)
    f.write('{0:d}'.format(height) + delimiter)
    for index in range(0, arr_of_lengths.shape[0] - 1): # all lengths except last
      f.write('{0:d}'.format(arr_of_lengths[index]) + delimiter)
    f.write('{0:d}'.format(arr_of_lengths[arr_of_lengths.shape[0] - 1])) # write last length without delimiter

  # calculate bits per pixel
  total_bits = 16 * (header_length_in_numbers + len(arr_of_lengths)) # 16 bits per word
  bits_per_pixel = total_bits / (width * height)
  return bits_per_pixel

with open("test.csv",mode='w') as f:
    f.write('{0:d}'.format(400) + ',')