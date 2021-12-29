import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
import torch

def botr_generator(dataset, annotations, dims, batch_size=4, fill_target=0.99, max_step_fill=0.1, step_fill_jitter = 0.3, for_nn=False):
  transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Resize(size=dims),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  while True:
    img_batch = []
    metadata_batch = []
    for b in range(batch_size):
      num_filled = 0
      composite_mask = torch.zeros((dims[0], dims[1], 1))
      composite = torch.zeros((dims[0], dims[1], 3))
      while num_filled / (dims[0] * dims[1]) < fill_target:
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
        fill_percent = torch.count_nonzero(fill_mask) / (dims[0] * dims[1])
        num_filled = torch.count_nonzero(composite_mask)
        this_step_fill = (torch.rand((1,)) * step_fill_jitter) + max_step_fill
        if fill_percent > this_step_fill:
          continue
        else:
          composite_mask = torch.logical_or(composite_mask, rand_mask)
          num_filled = torch.count_nonzero(composite_mask)
          composite += (fill_mask * img)
      composite = (composite - torch.min(composite)) / ((torch.max(composite) - torch.min(composite)) + 1e-7)
      if for_nn:
        composite = composite.permute(2, 0, 1)
      img_batch.append(composite)
      print(rand_ann)
      metadata = annotations[rand_ann["category_id"]]
      metadata_batch.append(metadata)
    img_batch = torch.stack(img_batch)
    metadata_batch = torch.stack(metadata_batch)
    yield img_batch, metadata_batch
