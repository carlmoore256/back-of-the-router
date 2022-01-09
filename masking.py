import numpy as np
from PIL import Image, ImageOps

def prepare_mask(annotation, dims, coco):
  mask = coco.annToMask(annotation)
  mask = Image.fromarray(mask)
  mask = ImageOps.fit(mask, dims[:2])
  mask = np.expand_dims(np.asarray(mask), -1) 
  mask = np.repeat(mask, 3, axis=-1)
  return mask

def create_exclusion_mask(mask_A, mask_B, format_uint8=True):
  excl_mask = mask_A.copy()
  excl_mask[mask_B > 0] = 0
  if format_uint8:
    excl_mask = np.uint8(np.clip((excl_mask) * 255, 0, 255))
  else:
    excl_mask = np.clip(excl_mask, 0, 1)    
  return excl_mask