"""
::

  Face averager

  Usage:
    averager.py --images=<images_folder> [--blur] [--plot]
              [--background=(black|transparent|average)]
              [--width=<width>] [--height=<height>]
              [--out=<filename>] [--destimg=<filename>]

  Options:
    -h, --help             Show this screen.
    --images=<folder>      Folder to images (.jpg, .jpeg, .png)
    --blur                 Flag to blur edges of image [default: False]
    --width=<width>        Custom width of the images/video [default: 500]
    --height=<height>      Custom height of the images/video [default: 600]
    --out=<filename>       Filename to save the average face [default: result.png]
    --destimg=<filename>   Destination face image to overlay average face
    --plot                 Flag to display the average face [default: False]
    --background=<bg>      Background of image to be one of (black|transparent|average) [default: black]
    --version              Show version.
"""

from docopt import docopt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from facemorpher import locator
from facemorpher import aligner
from facemorpher import warper
from facemorpher import blender
from facemorpher import plotter

def list_imgpaths(imgfolder):
  for fname in os.listdir(imgfolder):
    if (fname.lower().endswith('.jpg') or
       fname.lower().endswith('.png') or
       fname.lower().endswith('.jpeg')):
      yield os.path.join(imgfolder, fname)

def sharpen(img):
  blured = cv2.GaussianBlur(img, (0, 0), 2.5)
  return cv2.addWeighted(img, 1.4, blured, -0.4, 0)

def load_image_points(path, size):
  img = cv2.imread(path)
  points = locator.face_points(img)

  if len(points) == 0:
    print('No face in %s' % path)
    return None, None
  else:
    return aligner.resize_align(img, points, size)

def averager(imgpaths, dest_filename=None, width=500, height=600, background='black',
             blur_edges=False, out_filename='result.png', plot=False):

  size = (height, width)

  images = []
  point_set = []
  for path in imgpaths:
    img, points = load_image_points(path, size)
    if img is not None:
      images.append(img)
      point_set.append(points)

  if len(images) == 0:
    raise FileNotFoundError('Could not find any valid images.' +
                            ' Supported formats are .jpg, .png, .jpeg')

  if dest_filename is not None:
    dest_img, dest_points = load_image_points(dest_filename, size)
    if dest_img is None or dest_points is None:
      raise Exception('No face or detected face points in dest img: ' + dest_filename)
  else:
    dest_img = np.zeros(images[0].shape, np.uint8)
    dest_points = locator.average_points(point_set)

  num_images = len(images)
  result_images = np.zeros(images[0].shape, np.float32)
  for i in range(num_images):
    result_images += warper.warp_image(images[i], point_set[i],
                                       dest_points, size, np.float32)

  result_image = np.uint8(result_images / num_images)
  face_indexes = np.nonzero(result_image)
  dest_img[face_indexes] = result_image[face_indexes]

  mask = blender.mask_from_points(size, dest_points)
  if blur_edges:
    blur_radius = 10
    mask = cv2.blur(mask, (blur_radius, blur_radius))

  if background in ('transparent', 'average'):
    dest_img = np.dstack((dest_img, mask))

    if background == 'average':
      average_background = locator.average_points(images)
      dest_img = blender.overlay_image(dest_img, mask, average_background)

  print('Averaged {} images'.format(num_images))
  plt = plotter.Plotter(plot, num_images=1, out_filename=out_filename)
  plt.save(dest_img)
  plt.plot_one(dest_img)
  plt.show()

def main():
  args = docopt(__doc__, version='Face Averager 1.0')
  try:
    averager(list_imgpaths(args['--images']), args['--destimg'],
             int(args['--width']), int(args['--height']),
             args['--background'], args['--blur'], args['--out'], args['--plot'])
  except Exception as e:
    print(e)


if __name__ == "__main__":
  main()
