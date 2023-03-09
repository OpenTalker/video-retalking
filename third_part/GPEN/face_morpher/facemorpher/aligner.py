"""
Align face and image sizes
"""
import cv2
import numpy as np

def positive_cap(num):
  """ Cap a number to ensure positivity

  :param num: positive or negative number
  :returns: (overflow, capped_number)
  """
  if num < 0:
    return 0, abs(num)
  else:
    return num, 0

def roi_coordinates(rect, size, scale):
  """ Align the rectangle into the center and return the top-left coordinates
  within the new size. If rect is smaller, we add borders.

  :param rect: (x, y, w, h) bounding rectangle of the face
  :param size: (width, height) are the desired dimensions
  :param scale: scaling factor of the rectangle to be resized
  :returns: 4 numbers. Top-left coordinates of the aligned ROI.
    (x, y, border_x, border_y). All values are > 0.
  """
  rectx, recty, rectw, recth = rect
  new_height, new_width = size
  mid_x = int((rectx + rectw/2) * scale)
  mid_y = int((recty + recth/2) * scale)
  roi_x = mid_x - int(new_width/2)
  roi_y = mid_y - int(new_height/2)

  roi_x, border_x = positive_cap(roi_x)
  roi_y, border_y = positive_cap(roi_y)
  return roi_x, roi_y, border_x, border_y

def scaling_factor(rect, size):
  """ Calculate the scaling factor for the current image to be
      resized to the new dimensions

  :param rect: (x, y, w, h) bounding rectangle of the face
  :param size: (width, height) are the desired dimensions
  :returns: floating point scaling factor
  """
  new_height, new_width = size
  rect_h, rect_w = rect[2:]
  height_ratio = rect_h / new_height
  width_ratio = rect_w / new_width
  scale = 1
  if height_ratio > width_ratio:
    new_recth = 0.8 * new_height
    scale = new_recth / rect_h
  else:
    new_rectw = 0.8 * new_width
    scale = new_rectw / rect_w
  return scale

def resize_image(img, scale):
  """ Resize image with the provided scaling factor

  :param img: image to be resized
  :param scale: scaling factor for resizing the image
  """
  cur_height, cur_width = img.shape[:2]
  new_scaled_height = int(scale * cur_height)
  new_scaled_width = int(scale * cur_width)

  return cv2.resize(img, (new_scaled_width, new_scaled_height))

def resize_align(img, points, size):
  """ Resize image and associated points, align face to the center
    and crop to the desired size

  :param img: image to be resized
  :param points: *m* x 2 array of points
  :param size: (height, width) tuple of new desired size
  """
  new_height, new_width = size

  # Resize image based on bounding rectangle
  rect = cv2.boundingRect(np.array([points], np.int32))
  scale = scaling_factor(rect, size)
  img = resize_image(img, scale)

  # Align bounding rect to center
  cur_height, cur_width = img.shape[:2]
  roi_x, roi_y, border_x, border_y = roi_coordinates(rect, size, scale)
  roi_h = np.min([new_height-border_y, cur_height-roi_y])
  roi_w = np.min([new_width-border_x, cur_width-roi_x])

  # Crop to supplied size
  crop = np.zeros((new_height, new_width, 3), img.dtype)
  crop[border_y:border_y+roi_h, border_x:border_x+roi_w] = (
     img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w])

  # Scale and align face points to the crop
  points[:, 0] = (points[:, 0] * scale) + (border_x - roi_x)
  points[:, 1] = (points[:, 1] * scale) + (border_y - roi_y)

  return (crop, points)
