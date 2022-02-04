from PIL import ImageFilter, Image
from config import DATASET_CONFIG
import os
import numpy as np
from skimage.exposure import equalize_adapthist, histogram

def sharpen_image(image, loops=1):
  for l in range(loops):
    image = image.filter(ImageFilter.SHARPEN)
  return image

def jpeg_decimation(image, quality=10, loops=10, numpy=False):
  temp_file = os.path.join(DATASET_CONFIG["temp_path"],'decimate.jpg')
  if numpy:
    image = Image.fromarray(np.clip(image.astype(np.uint8), 0, 255))
  for i in range(loops):
    image.save(temp_file, quality=quality)
    image = Image.open(temp_file)
  if numpy:
    image = np.asarray(image)
  return image

def adaptive_hist(image, kernel_size=None, clip_limit=0.5, numpy=True):
  if numpy:
    image = image.astype(np.float) / 255.
  if kernel_size is None:
    kernel_size = int(image.shape[0] * 1/8)
  image = equalize_adapthist(image, kernel_size, clip_limit)
  if numpy:
    image = np.clip((image * 255).astype(np.uint8), 0, 255)
  return image

def image_histogram(image, config=None):
  if config is None:
    config={'normalizeHistogram' : False, 'histogramChannelAxis' : -1}
  try:
    hist = histogram(image, 
                    normalize=config['normalizeHistogram'],
                    channel_axis=config['histogramChannelAxis'])[0]
  except Exception as e:
      print(f'[!] Histogram error: {e}')
      return None
  # add some handing of different sizes produced by histogram for whatever reason
  if hist.shape[-1] < image.shape[0]:
    hist = np.pad(hist, ((0,0),(0, image.shape[0] - hist.shape[-1])))
  elif hist.shape[-1] > image.shape[0]:
    hist = hist[:, :image.shape[0]]
  return hist

