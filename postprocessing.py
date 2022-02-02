from PIL import ImageFilter, Image
from config import DATASET_CONFIG
import os

def sharpen_image(image, loops=1):
  for l in range(loops):
    image = image.filter(ImageFilter.SHARPEN)
  return image

def jpeg_decimation(image, quality=10, loops=10):
  temp_file = os.path.join(DATASET_CONFIG["temp_path"],'decimate.jpg')
  for i in range(loops):
    image.save(temp_file, quality=quality)
    image = Image.open(temp_file)
  return image
