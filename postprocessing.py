from PIL import ImageFilter

def sharpen_image(image, loops=1):
  for l in range(loops):
    image = image.filter(ImageFilter.SHARPEN)
  return image