import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import torchvision.transforms.functional as TF
import random
import scipy.ndimage

def decimate(image, kernel):
  # blurred = scipy.signal.convolve2d(image, kernel, mode="same")
  blurred = scipy.ndimage.filters.convolve(image, kernel, mode="constant")
  return blurred[::2, ::2]

def interpolate(image, kernel):
  img_up = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
  img_up[::2, ::2] = image
  return scipy.ndimage.filters.convolve(img_up, kernel * 4, mode="constant")

def create_kernel():
  return np.array([[1,4,6,4,1],
                   [4,16,24,16,4],
                   [6,24,36,24,6],
                   [4,16,24,16,4],
                   [1,4,6,4,1]]) * (1/256)


kernel = create_kernel()

# the way shown in the slides (better)
def generate_pyramid(image, kernel):
  G = [image,]
  L = []

  while image.shape[0] >- 2 and image.shape[1] >= 2:
    image = decimate(image, kernel)
    G.append(image)

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
def blend_masked(img_A, img_B, mask_A, mask_B, plot_levels=False):
  kernel = create_kernel()


  img1_G, img1_L = generate_pyramid(img_A, kernel)
  img2_G, img2_L = generate_pyramid(img_B, kernel)

  blended_L = []
  # reverse order of pyramid
  img1_L = img1_L[::-1]
  img2_L = img2_L[::-1]
  # iterate through levels and blend spatial frequencies
  for l_1, l_2 in zip(img1_L, img2_L):
    mask_A_scaled = torch.as_tensor(mask_A).permute(2, 0, 1)
    mask_B_scaled = torch.as_tensor(mask_B).permute(2, 0, 1)

    mask_A_scaled = TF.resize(mask_A_scaled, size=(l_1.shape[0],l_1.shape[1])).squeeze(0).numpy().astype(float)
    mask_B_scaled = TF.resize(mask_B_scaled, size=(l_2.shape[0], l_2.shape[1])).squeeze(0).numpy().astype(float)

    # mask_A_scaled = scipy.ndimage.filters.convolve(mask_A_scaled, kernel, mode="constant") 
    # mask_B_scaled = scipy.ndimage.filters.convolve(mask_B_scaled, kernel, mode="constant")

    mask_A_scaled = np.clip(mask_A_scaled, 0., 1.)
    mask_B_scaled = np.clip(mask_B_scaled, 0., 1.)

    l_1 *= mask_A_scaled
    l_2 *= mask_B_scaled
    
    blend = l_1+l_2
    # blend = (blend - np.min(blend)) / ((np.max(blend) - np.min(blend)) + 1e-7)
    

    blended_L.append(blend)

    if plot_levels:
      plt.subplot(151)
      plt.title("img A")
      plt.imshow(l_1, cmap="gray")
      plt.subplot(152)
      plt.title("img B")
      plt.imshow(l_2, cmap="gray")
      plt.subplot(153)
      plt.title("mask A")
      plt.imshow(mask_A_scaled, cmap="gray")
      plt.subplot(154)
      plt.title("mask B")
      plt.imshow(mask_B_scaled, cmap="gray")
      plt.subplot(155)
      plt.title("blended")
      plt.imshow(blend, cmap="gray")
      plt.show()
 
  blended_L = blended_L[::-1]

  recon = reconstruct(img1_G[-1], blended_L, kernel)
  return recon


def blend_masked_rgb(img1_c, img2_c, mask_A, mask_B):
  r = blend_masked(img1_c[:,:,0], img2_c[:,:,0], mask_A, mask_B, False)
  g = blend_masked(img1_c[:,:,1], img2_c[:,:,1], mask_A, mask_B)
  b = blend_masked(img1_c[:,:,2], img2_c[:,:,2], mask_A, mask_B)

  blended_c = np.stack((r,g,b), axis=-1)
  return blended_c