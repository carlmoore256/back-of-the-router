import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pyramids import blend_masked_rgb, sharpen
import numpy as np
import random
import torch


def generate_name(metadata):
  word_len = np.random.randint(3, 20)
  sorted_attrs = dict(sorted(metadata.items(), key=lambda item: item[1], reverse=True))
  name = ""
  attr_idx = 0
  while len(name) < word_len:
      print(attr_idx)
      key = list(sorted_attrs.keys())[attr_idx]
      slice_len = int((sorted_attrs[key] * word_len) ** 2)
      if slice_len == 0:
          slice_len = 1
      name += key[:slice_len]
      attr_idx += 1
  return name

def generate_single(dataset, info, dims, fill_target=0.99, max_step_fill=0.1, step_fill_jitter = 0.3, sharpness=10, for_nn=False):
  transform = transforms.Compose(
  [transforms.ToTensor(), 
  transforms.Resize(size=dims),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  unique_categories = list(set([k["supercategory"] for k in info['categories']]))

  while True:
    total_px_filled = 0
    composite_mask = torch.zeros((dims[0], dims[1], 1))
    composite = torch.zeros((dims[0], dims[1], 3))
    metadata = {}
    metadata = {key: 0 for key in unique_categories}
    map = {}
    # name = ""

    for cat in info['categories']:
        map[cat['id']] = cat["supercategory"]

    while total_px_filled / (dims[0] * dims[1]) < fill_target:
      idx = np.random.randint(0, dataset.__len__())
      img, ann = dataset[idx]
      

      rand_ann = random.choice(ann)
      img = TF.resize(img, size=dims)
      img = img.permute(1,2,0)
      rand_mask = dataset.coco.annToMask(rand_ann)
      rand_mask = torch.as_tensor(rand_mask).unsqueeze(0)
      rand_mask = TF.resize(rand_mask, size=(dims[0], dims[1]))
      rand_mask = torch.permute(rand_mask,(1,2,0))
      px_filled = torch.count_nonzero(rand_mask).item()
      fill_percent = px_filled / (dims[0] * dims[1])
      # masked = img * rand_mask
      category = map[rand_ann['category_id']]
      fill_mask = torch.zeros_like(composite_mask)
      fill_mask = torch.logical_or(rand_mask, composite_mask)
      fill_mask[composite_mask == 1] = 0
      px_filled = torch.count_nonzero(fill_mask).item()
      fill_percent = px_filled / (dims[0] * dims[1])
      this_step_fill = (torch.rand((1,)) * step_fill_jitter) + max_step_fill

      if fill_percent > this_step_fill or fill_percent == 0:
        continue
      else:
        category = map[rand_ann['category_id']]
        metadata[category] += fill_percent
        composite_mask = torch.logical_or(composite_mask, rand_mask)

        composite = blend_masked_rgb(
          img_A=img.numpy(), # use the image so the blending works with a wider area
          img_B=composite.numpy(), 
          mask_A=fill_mask.numpy(), 
          mask_B=composite_mask.numpy(),
          kernel_size=1,
          kernel_sig=1)

        composite = torch.as_tensor(composite)
        # composite += fill_mask * img
        total_px_filled = torch.count_nonzero(composite_mask)
        print(f'px filled {total_px_filled} fill percent {total_px_filled/(dims[0]*dims[1])}')
        letters_category = int(((len(category)) * fill_percent)**4)
        # name += category[:letters_category]

    composite = sharpen(composite.permute(2, 0, 1), sharpness).permute(1, 2, 0)

    composite = (composite - torch.min(composite)) / ((torch.max(composite) - torch.min(composite)))
    if for_nn:
      composite = composite.permute(2, 0, 1)

    name = generate_name(metadata)
    yield composite, metadata, name

def botr_generator(dataset, annotations, dims, batch_size=4, fill_target=0.99, max_step_fill=0.1, step_fill_jitter = 0.3, for_nn=False):
  transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Resize(size=dims),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  while True:
    img_batch = []
    metadata_batch = []
    b = 0
    batch_ready = False
    while not batch_ready:
    # for b in range(batch_size):
      px_filled = 0
      composite_mask = torch.zeros((dims[0], dims[1], 1))
      composite = torch.zeros((dims[0], dims[1], 3))
      attr_list = []
      while px_filled / (dims[0] * dims[1]) < fill_target:
        idx = np.random.randint(0, dataset.__len__())
        img, ann = dataset[idx]
        rand_ann = random.choice(ann)
        img = TF.resize(img, size=dims)
        img = img.permute(1,2,0)
        rand_mask = dataset.coco.annToMask(rand_ann)
        rand_mask = torch.as_tensor(rand_mask).unsqueeze(0)
        rand_mask = TF.resize(rand_mask, size=(dims[0], dims[1]))
        rand_mask = torch.permute(rand_mask,(1,2,0))
        fill_mask = torch.zeros_like(composite_mask)
        fill_mask = torch.logical_or(rand_mask, composite_mask)
        fill_mask[composite_mask == 1] = 0

        px_filled = torch.count_nonzero(composite_mask)
        fill_percent = px_filled / (dims[0] * dims[1])

        this_step_fill = (torch.rand((1,)) * step_fill_jitter) + max_step_fill
        if fill_percent > this_step_fill:
          continue
        else:
          attr_list.append(ann)
          # composite = blend_masked_rgb(composite.numpy(), img.numpy(), composite_mask.numpy(), fill_mask.numpy())
          # composite = torch.as_tensor(composite)
          # composite_mask = torch.logical_or(composite_mask, rand_mask)

          px_filled = torch.count_nonzero(composite_mask)
          composite_mask = torch.logical_or(composite_mask, rand_mask)
          # px_filled = torch.count_nonzero(composite_mask)
          composite += (fill_mask * img)
      composite = (composite - torch.min(composite)) / ((torch.max(composite) - torch.min(composite)) + 1e-7)
      if for_nn:
        composite = composite.permute(2, 0, 1)

      # FIX SO WHOLE ATTR LIST IS PUT INTO METADATA LIST
      # if rand_ann["category_id"] - 100 in annotations.keys():
      img_batch.append(composite)
      # metadata = annotations[rand_ann["category_id"] - 100]
      # metadata = annot
      metadata_batch.append(metadata)
      b += 1
      if b == batch_size:
        batch_ready = True
    # else:
    #   print(f"{rand_ann['category_id'] - 100} \n")

    img_batch = torch.stack(img_batch)
    yield img_batch, metadata_batch
