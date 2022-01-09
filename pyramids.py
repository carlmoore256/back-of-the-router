import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import convolve2d
import torch
import torchvision.transforms.functional as TF
import random
import scipy.ndimage
# from PIL import Image
import cv2

def sharpen(image, amount):
  image = TF.adjust_sharpness(image, amount)
  return image

def decimate(image, kernel):
  # blurred = scipy.signal.convolve2d(image, kernel, mode="same")
  # blurred = scipy.ndimage.filters.convolve(image, kernel, mode="constant")
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


# input two single channel images img_A, img_B and a mask
# blend pyramids with provided mask generator function mask_f
# provide gaussian kernel
def blend_masked(img_A, img_B, mask_A, mask_B, kernel_size=5, kernel_sig=1, max_depth=3, plot_levels=False):
  kernel = create_kernel(kernel_size, kernel_sig)
  img1_G, img1_L = generate_pyramid(img_A, kernel, max_depth)
  img2_G, img2_L = generate_pyramid(img_B, kernel, max_depth)

  blended_L = []
  # reverse order of pyramid
  img1_L = img1_L[::-1]
  img2_L = img2_L[::-1]

  # iterate through levels and blend spatial frequencies
  for l_1, l_2 in zip(img1_L, img2_L):
    mask_A_scaled = cv2.resize(mask_A, (l_1.shape[0], l_1.shape[1]))
    mask_B_scaled = cv2.resize(mask_B, (l_2.shape[0], l_2.shape[1]))

    mask_A_scaled = np.clip(mask_A_scaled, 0., 1.)
    mask_B_scaled = np.clip(mask_B_scaled, 0., 1.)

    l_1 *= mask_A_scaled
    l_2 *= mask_B_scaled
    
    blend = l_1+l_2
    # blend = (blend - np.min(blend)) / ((np.max(blend) - np.min(blend)) + 1e-7)
    blended_L.append(blend)

    if plot_levels:
      plot_pyramid_level(l_1, l_2, mask_A_scaled, mask_B_scaled, blend)
 
  blended_L = blended_L[::-1]
  recon = reconstruct(img1_G[-1], blended_L, kernel)
  return recon

def plot_pyramid_level(l_1, l_2, mask_A, mask_B, blend):
  plt.subplot(151)
  plt.title("img A")
  plt.imshow(l_1, cmap="gray")
  plt.subplot(152)
  plt.title("img B")
  plt.imshow(l_2, cmap="gray")
  plt.subplot(153)
  plt.title("mask A")
  plt.imshow(mask_A, cmap="gray")
  plt.subplot(154)
  plt.title("mask B")
  plt.imshow(mask_B, cmap="gray")
  plt.subplot(155)
  plt.title("blended")
  plt.imshow(blend, cmap="gray")
  plt.show()

def blend_masked_rgb(img_A, img_B, mask_A, mask_B, kernel_size=5, kernel_sig=1, max_depth=None):
  blended_rgb = []
  
  for channel in range(3):
    blended = blend_masked(
        img_A[:,:,channel], img_B[:,:,channel], mask_A, mask_B, kernel_size, kernel_sig, max_depth)  
    blended_rgb.append(blended)

  # r = blend_masked(img_A[:,:,0], img_B[:,:,0], mask_A, mask_B, kernel_size=kernel_size, kernel_sig=kernel_sig, max_depth=max_depth)
  # g = blend_masked(img_A[:,:,1], img_B[:,:,1], mask_A, mask_B, kernel_size=kernel_size, kernel_sig=kernel_sig, max_depth=max_depth)
  # b = blend_masked(img_A[:,:,2], img_B[:,:,2], mask_A, mask_B, kernel_size=kernel_size, kernel_sig=kernel_sig, max_depth=max_depth)
  blended_rgb = np.stack(blended_rgb, axis=-1)
  return blended_rgb



      # mask_A_scaled = torch.as_tensor(mask_A).permute(2, 0, 1)
    # mask_B_scaled = torch.as_tensor(mask_B).permute(2, 0, 1)
    # mask_A_scaled = TF.resize(mask_A_scaled, size=(l_1.shape[0],l_1.shape[1])).squeeze(0).numpy().astype(float)
    # mask_B_scaled = TF.resize(mask_B_scaled, size=(l_2.shape[0], l_2.shape[1])).squeeze(0).numpy().astype(float)
    # mask_A_scaled = scipy.ndimage.filters.convolve(mask_A_scaled, kernel, mode="constant") 
    # mask_B_scaled = scipy.ndimage.filters.convolve(mask_B_scaled, kernel, mode="constant")