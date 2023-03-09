"""
Plot and save images
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path
import numpy as np
import cv2

def bgr2rgb(img):
  # OpenCV's BGR to RGB
  rgb = np.copy(img)
  rgb[..., 0], rgb[..., 2] = img[..., 2], img[..., 0]
  return rgb

def check_do_plot(func):
  def inner(self, *args, **kwargs):
    if self.do_plot:
      func(self, *args, **kwargs)

  return inner

def check_do_save(func):
  def inner(self, *args, **kwargs):
    if self.do_save:
      func(self, *args, **kwargs)

  return inner

class Plotter(object):
  def __init__(self, plot=True, rows=0, cols=0, num_images=0, out_folder=None, out_filename=None):
    self.save_counter = 1
    self.plot_counter = 1
    self.do_plot = plot
    self.do_save = out_filename is not None
    self.out_filename = out_filename
    self.set_filepath(out_folder)

    if (rows + cols) == 0 and num_images > 0:
      # Auto-calculate the number of rows and cols for the figure
      self.rows = np.ceil(np.sqrt(num_images / 2.0))
      self.cols = np.ceil(num_images / self.rows)
    else:
      self.rows = rows
      self.cols = cols

  def set_filepath(self, folder):
    if folder is None:
      self.filepath = None
      return

    if not os.path.exists(folder):
      os.makedirs(folder)
    self.filepath = os.path.join(folder, 'frame{0:03d}.png')
    self.do_save = True

  @check_do_save
  def save(self, img, filename=None):
    if self.filepath:
      filename = self.filepath.format(self.save_counter)
      self.save_counter += 1
    elif filename is None:
      filename = self.out_filename

    mpimg.imsave(filename, bgr2rgb(img))
    print(filename + ' saved')

  @check_do_plot
  def plot_one(self, img):
    p = plt.subplot(self.rows, self.cols, self.plot_counter)
    p.axes.get_xaxis().set_visible(False)
    p.axes.get_yaxis().set_visible(False)
    plt.imshow(bgr2rgb(img))
    self.plot_counter += 1

  @check_do_plot
  def show(self):
    plt.gcf().subplots_adjust(hspace=0.05, wspace=0,
                              left=0, bottom=0, right=1, top=0.98)
    plt.axis('off')
    #plt.show()
    plt.savefig('result.png')

  @check_do_plot
  def plot_mesh(self, points, tri, color='k'):
    """ plot triangles """
    for tri_indices in tri.simplices:
      t_ext = [tri_indices[0], tri_indices[1], tri_indices[2], tri_indices[0]]
      plt.plot(points[t_ext, 0], points[t_ext, 1], color)
