import numpy as np
from scipy.signal import convolve2d
from utils import display_multiple_images
import cv2

def decimate(image, kernel):
  decimated = convolve2d(image, kernel, mode='same', boundary='wrap')
  return decimated[::2, ::2]

def interpolate(image, kernel):
  img_up = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
  img_up[::2, ::2] = image
  return convolve2d(img_up, kernel, mode="same", boundary='wrap')

# creates gaussian kernel with side length `l` and a sigma of `sig`
def create_kernel(l=5, sig=1.):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def generate_pyramid(image, kernel, max_depth=None):
  G = [image,]
  L = []

  depth = 0

  while image.shape[0] > 2 and image.shape[1] > 2:
    if max_depth is not None and depth == max_depth:
      break
    image = decimate(image, kernel)
    G.append(image)
    depth += 1

  for i in range(len(G)-1):
    L.append(G[i] - interpolate(G[i+1], kernel))

  return G[:-1], L

# gaussian is the final image in the gaussian pyramid
# laplacian is the list of laplacian images
def reconstruct(gaussian, laplacian, kernel):
  recovered = gaussian
  L_up = laplacian[::-1][1:]
  for l in L_up:
    recovered = interpolate(recovered, kernel) + l
  return recovered
def dilate_mask(mask, kernel, iter=1):
  dilated = cv2.dilate(mask, kernel, iterations = iter)
  return dilated

def blur_mask(mask, kernel, steps=1):
  for _ in range(steps):
    mask = convolve2d(
      mask, kernel, mode='same', boundary='fill')
  mask = np.clip(mask, 0., 1.)
  mask = np.nan_to_num(mask)
  return mask

def blend_masked(
  img_A, img_B, mask_A, mask_B,
  blendConfig):

  pyrKernel = create_kernel(
    blendConfig['pyr_kernel_size'], blendConfig['pyr_kernel_sigma'])

  maskKernel = create_kernel(
    blendConfig['mask_kernel_size'], blendConfig['mask_kernel_sigma'])

  img1_G, img1_L = generate_pyramid(img_A, pyrKernel, blendConfig['max_depth'])
  _,      img2_L = generate_pyramid(img_B, pyrKernel, blendConfig['max_depth'])

  blended_L = []
  # reverse order of pyramid
  img1_L = img1_L[::-1]
  img2_L = img2_L[::-1]

  # iterate through levels and blend spatial frequencies
  for l_1, l_2 in zip(img1_L, img2_L):

    mask_A_scaled = cv2.resize(mask_A, (l_1.shape[0], l_1.shape[1]))
    mask_B_scaled = cv2.resize(mask_B, (l_2.shape[0], l_2.shape[1]))
    
    if blendConfig['blur_masks']:
      mask_A_scaled = blur_mask(mask_A_scaled, maskKernel, blendConfig['blur_iters'])
      mask_B_scaled = blur_mask(mask_B_scaled, maskKernel, blendConfig['blur_iters'])

    l_1 *= mask_A_scaled
    l_2 *= mask_B_scaled
    
    blend = l_1+l_2
    blended_L.append(blend)

    if blendConfig['plot_levels']:
      display_multiple_images([l_1, l_2, mask_A_scaled, mask_B_scaled, blend],
                      ['l_1', 'l_2', 'mask_A', 'mask_B', 'blended'])
  
  blended_L = blended_L[::-1]
  recon = reconstruct(img1_G[-1], blended_L, pyrKernel)
  return recon

def normalize_values(values):
  values = (values - np.min(values)) / (np.max(values) - np.min(values))
  return values

# convert float 0.-1. to  uint8 0-255
def to_uint8_range(values):
  values = np.clip(values * 255, 0, 255)
  values = values.astype(np.uint8)
  return values

# convert uint8 0-255 to 0.-1.
def to_float_range(values):
  values = values.astype(float)
  values = np.clip(values / 255, 0., 1.)
  return values

def blend_masked_rgb(img_A, img_B, 
  mask_A, mask_B, blendConfig, plotLevels=False):

  img_A = to_float_range(img_A)
  img_B = to_float_range(img_B)
  blended_rgb = []
  for channel in range(3):
    blended = blend_masked(
        img_A[:,:,channel], img_B[:,:,channel], 
        mask_A, mask_B, blendConfig)
    blended_rgb.append(blended)
  blended_rgb = np.stack(blended_rgb, axis=-1)
  blended_rgb = normalize_values(blended_rgb)
  blended_rgb = to_uint8_range(blended_rgb)
  return blended_rgb