import numpy as np
from PIL import Image, ImageOps
from utils import arr2d_to_3d

def resize_fit(array, dims):
    fit = Image.fromarray(array)
    fit = np.asarray(ImageOps.fit(fit, dims))
    if len(fit.shape) == 2:
      fit = np.expand_dims(fit, -1)
    return fit

def create_exclusion_mask(mask_A, mask_B, format_uint8=True):
  excl_mask = mask_A.copy()
  excl_mask[mask_B > 0] = 0
  if format_uint8:
    excl_mask = np.uint8(np.clip((excl_mask) * 255, 0, 255))
  else:
    excl_mask = np.clip(excl_mask, 0, 1)    
  return excl_mask

def mask_add_composite(source, mask, composite):
    mask = np.clip(mask, 0, 1)
    masked = np.uint8(source * mask)
    # composite[exclusionMask > 0] += np.clip(image * exclusionMask, 0, 255).astype(np.uint8)
    composite = np.clip(composite + masked, 0, 255)
    return composite

def calc_fill_percent(mask, area):
    return np.count_nonzero(mask) / float(area)

# def add_to_composite(mask)
    # source = Image.fromarray(source)
    # mask = Image.fromarray(mask)
    # composite = Image.fromarray(composite)
    # composite = Image.composite(source, composite, mask)