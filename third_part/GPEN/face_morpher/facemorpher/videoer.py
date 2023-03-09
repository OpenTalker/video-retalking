"""
Create a video with image frames
"""

import cv2
import numpy as np


def check_write_video(func):
  def inner(self, *args, **kwargs):
    if self.video:
      return func(self, *args, **kwargs)
    else:
      pass
  return inner


class Video(object):
  def __init__(self, filename, fps, w, h):
    self.filename = filename

    if filename is None:
      self.video = None
    else:
      fourcc = cv2.VideoWriter_fourcc(*'MJPG')
      self.video = cv2.VideoWriter(filename, fourcc, fps, (w, h), True)

  @check_write_video
  def write(self, img, num_times=1):
    for i in range(num_times):
      self.video.write(img[..., :3])

  @check_write_video
  def end(self):
    print(self.filename + ' saved')
    self.video.release()
