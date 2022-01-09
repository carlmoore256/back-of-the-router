import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pyramids import blend_masked_rgb, sharpen
from utils import load_dict, save_dict, filter_list
from coco_utils import generate_assets, coco_value_distribution, closest_sized_annotation
import numpy as np
import os
import random
import torch

# a class that can remain loaded, and change aspects
# of generative properties
class BOTR_Generator():

  def __init__(self, dataset_path="dataset/"):

    pathCatMap = os.path.join(dataset_path, "category_map.pickle")
    pathCocoOrganized = os.path.join(dataset_path, "coco_organized.pickle")
    if not os.path.isfile(pathCatMap) and not os.path.isfile(pathCocoOrganized):
      generate_assets(dataset_path)

    self.category_map = load_dict(pathCatMap)
    self.coco_examples = load_dict(pathCocoOrganized)
    self.all_ids = list(self.coco_examples.keys())

    self.distStuff, self.meanAreaStuff, self.stdAreaStuff = coco_value_distribution(
      self.coco_examples, key="stuff_ann", val_key="area")
  
    self.distInstances, self.meanAreaInstances, self.stdAreaInstances = coco_value_distribution(
      self.coco_examples, key="instance_ann", val_key="area")

  # get an example coco image, if no id is provided return a random one
  def get_coco_example(self, id=None):
    if id == None:
      id = random.choice(self.all_ids)
    return self.coco_examples[id]
 
  # def fill_with_annotation()

  # selects an average sized chunk of stuff or instances
  def random_size_target(self, annKey="stuff_ann", sizeScalar=1., stdScalar=1.):
    if annKey == "stuff_ann":
      loc = self.meanAreaStuff * sizeScalar
      scale = self.stdAreaStuff * stdScalar
    elif annKey == "instance_ann":
      loc = self.meanAreaInstances * sizeScalar
      scale = self.stdAreaInstances * stdScalar
    return np.random.normal(loc=loc, scale=scale, size=[1])
  
  # select an annotation from a gaussian distribution
  def gaussian_select_patch(self, annList, annKey, avgSize=1., sizeVariance=1.):
    sizeTarget = self.random_size_target(
      annKey, 
      sizeScalar=avgSize, 
      stdScalar=sizeVariance)
    return closest_sized_annotation(sizeTarget, annList)

  def gen_attribute_dict(self):
    attributes = {cat["supercategory"]: 0 for cat in self.category_map.values()}
    attributes['text_metadata'] = {
    "objects" : [],
    "descriptions" : [] }
    return attributes

  def ann_category_name(self, annotation):
    name = self.category_map[annotation["category_id"]]["supercategory"]
    return name

  def filter_ann_categories(self, annotations, allowedCategories):
    # map string category names to annotations
    annCategories = {self.ann_category_name(ann) : ann for ann in annotations}
    # construct list of allowed values
    filtered = [v for k, v in annCategories if k in allowedCategories]
    return filtered
    
  def generate_image(self, config):

    config = {
      # average size of each patch (1 being mean of distribution)
      'avgPatchSize' : 0.8,
      # average size variance of each patch added
      'avgPatchVariance' : 0.01,
      # target percentage of pixels to fill
      'targetFill' : 0.99,
      # output image size
      'outputSize' : (512, 512),
      # prevent supercategories from appearing
      'allowedCategories' : ["person", "other"],
      # choose either "stuff_ann" or "instances_ann"
      'ann_key' : "stuff_ann",
      # prevent supercategories from appearing
      'disallowed_catg' : ["person", "other"],
      # choose either "stuff_ann" or "instances_ann"
    }
 

    composite = np.zeros(
      (config["outputSize"][0], config["outputSize"][1], 3),
      dtype=np.uint8)

    attributes = self.gen_attribute_dict()

    px_filled = 0

    while px_filled < config['targetFill']:
      randCoco = self.get_coco_example()
      annList = randCoco[config['ann_key']]

      if len(annList) == 0:
        print(f'! len of annotation is 0, skipping')
        continue

      annList = self.filter_ann_categories(annList, config["allowedCategories"])

      # randomly select a coco annotation with
      # gaussian probability of selecting an average sized one
      chosenAnn = self.gaussian_select_patch(
        annList=randCoco[config['ann_key']], 
        annKey=config['ann_key'],
        avgSize=config['avgPatchSize'],
        avgVariance=['avgPatchVariance'])

      annCateg = self.ann_category_name(chosenAnn)

      if annCateg not in config["allowedCategories"]:
        print(f'{annCateg} not in allowedCategories, skipping...')
        continue






# def attribute_map(category_map)
# attributes = {cat["supercategory"]: 0 for cat in category_map.values()}

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

def generate_single(
  dataset, info, dims, 
  fill_target=0.99, 
  max_step_fill=0.1, 
  step_fill_jitter = 0.3, 
  sharpness=10, 
  kernel_size=1,
  kernel_sig=1, 
  for_nn=False):
  
  transform = transforms.Compose(
  [transforms.ToTensor(), 
  transforms.Resize(size=dims),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  unique_categories = list(set([k["supercategory"] for k in info['categories']]))

  while True:
    total_px_filled = 0
    composite_mask = torch.zeros((dims[0], dims[1], 1))
    composite = torch.zeros((dims[0], dims[1], 3))
    # metadata = {}
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
          kernel_size=kernel_size,
          kernel_sig=kernel_sig)

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
    metadata["name"] = name
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
