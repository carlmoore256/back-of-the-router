# from PIL.Image import composite, fromarray
from PIL import Image
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
from postprocessing import sharpen_image
from pyramids import blend_masked_rgb
from utils import load_dict, save_dict, filter_list, display_multiple_images, save_asset_metadata_pair
from coco_utils import load_coco_info, load_coco_image, generate_assets, coco_value_distribution, closest_sized_annotation, print_generator_status, load_coco_obj
from masking import resize_fit, create_exclusion_mask, mask_add_composite, calc_fill_percent
from wording import generate_name, zipf_description
import numpy as np
import os
import random
# import torch
from tqdm import tqdm

# a class that can remain loaded, and change aspects
# of generative properties
class BOTR_Generator():
  # ADD METHOD TO FILTER OUT VALID LICENSES!
  def __init__(self, dataset_path="dataset/"):

    pathCatMap = os.path.join(dataset_path, "category_map.pickle")
    pathCocoOrganized = os.path.join(dataset_path, "coco_organized.pickle")
    if not os.path.isfile(pathCatMap) and not os.path.isfile(pathCocoOrganized):
      generate_assets(dataset_path)
    
    print(f'loading coco assets - {pathCatMap}')
    self.category_map = load_dict(pathCatMap)
    print(f'loading coco assets - {pathCocoOrganized}')
    self.coco_examples = load_dict(pathCocoOrganized)
    self.all_ids = list(self.coco_examples.keys())

    captionsPath = os.path.join(dataset_path, "annotations", "captions_train2017.json")
    print(f'loading coco assets - {captionsPath}')
    self.cocoCaptions = load_coco_obj(captionsPath)

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

  # use the coco object to extract a binary mask
  def get_mask(self, ann):
    return self.cocoCaptions.annToMask(ann)

  # selects an average sized chunk of stuff or instances
  def random_size_target(self, annKey="stuff_ann", sizeScalar=1., stdScalar=1.):
    if annKey == "stuff_ann":
      loc = self.meanAreaStuff * sizeScalar
      scale = self.stdAreaStuff * stdScalar
    elif annKey == "instance_ann":
      loc = self.meanAreaInstances * sizeScalar
      scale = self.stdAreaInstances * stdScalar
    elif annKey == "any":
      loc = ((self.meanAreaInstances + self.meanAreaStuff) / 2) * sizeScalar
      scale = ((self.stdAreaInstances + self.stdAreaStuff) / 2) * stdScalar
    return np.random.normal(loc=loc, scale=scale, size=[1])
  
  # select an annotation from a gaussian distribution
  def gaussian_select_patch(self, annList, annKey, avgSize=1., sizeVariance=1.):
    if len(annList) > 1:
      sizeTarget = self.random_size_target(
        annKey, 
        sizeScalar=avgSize, 
        stdScalar=sizeVariance)
      return closest_sized_annotation(annList, sizeTarget)
    else:
      return annList[0]

  def gen_attribute_dict(self):
    attributes = {"category_percentage": {cat["supercategory"]: 0 for cat in self.category_map.values()}}
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
    filtered = [v for k, v in annCategories.items() if k in allowedCategories]
    return filtered

  # extends functionality to retrive both stuff and instances
  def get_random_ann_list(self, ann_key):
    randCoco = self.get_coco_example()
    if ann_key == "any":
      annList = randCoco["stuff_ann"]
      annList += randCoco["instance_ann"]
      return annList
    else:
      return randCoco[ann_key]


  def generate_image(self, config, imageProgress=False, printWarnings=False):

    composite = np.zeros((config["outputSize"][0], config["outputSize"][1],3), dtype=np.uint8)
    compositeMask = np.zeros((config["outputSize"][0], config["outputSize"][1], 1), dtype=np.float)

    attributes = self.gen_attribute_dict()
    totalArea = config['outputSize'][0]*config['outputSize'][1]

    px_filled = 0
    prev_px_filled = 0
    skipped = 0

    # components = [] # save each component of the constructed image along the way

    pbar = tqdm(total=config['targetFill'])
    while px_filled < config['targetFill']:
      randCoco = self.get_coco_example()

      if config['ann_key'] == "any":
        annList = randCoco["stuff_ann"]
        annList += randCoco["instance_ann"]
      else:
        annList = randCoco[config['ann_key']]

      annList = self.get_random_ann_list(config['ann_key'])
      annList = self.filter_ann_categories(annList, config["allowedCategories"])
      
      if len(annList) == 0:
        if printWarnings:
          print(f'! len of valid annotations is 0, skipping...')
        skipped+=1
        continue
  
      # randomly select a coco annotation with
      # gaussian probability of selecting a given sized one
      chosenAnn = self.gaussian_select_patch(
        annList=annList,
        annKey=config['ann_key'],
        avgSize=config['avgPatchSize'],
        sizeVariance=config['avgPatchVariance'])

      # obtain mask of the coco annotation
      objectMask = self.get_mask(chosenAnn)
      # resize the mask to fit within bounds of output size
      # here we can choose to apply other transformations if we want
      objectMask = resize_fit(objectMask, config['outputSize'])
      exclusionMask = create_exclusion_mask(objectMask, compositeMask)
      areaPercent = calc_fill_percent(objectMask, totalArea)
      
      if not exclusionMask.any() or areaPercent < config['minPatchArea'] or areaPercent > config['maxPatchArea']:
        if printWarnings:
          print(f'! area {areaPercent}% does not match area parameters: min {config["minPatchArea"]} max {config["maxPatchArea"]}')
        skipped+=1
        continue

      image = load_coco_image(randCoco['filename'], fit=config['outputSize'])

      if config['image_blending']['use_blending']:
        blendedComposite = blend_masked_rgb(
          img_A=image,
          img_B=composite, 
          mask_A=exclusionMask, 
          mask_B=compositeMask,
          blendConfig=config['image_blending'])

        composite = blendedComposite
      else:
        # add the masked content on to the composite (inplace modify composite)
        composite = mask_add_composite(image, exclusionMask, composite)

      compositeMask = np.logical_or(exclusionMask, compositeMask).astype(np.uint8)
      
      if imageProgress:
        display_multiple_images(
          [image, objectMask, compositeMask, exclusionMask, composite], 
          ["original img", "chosen mask", "composite_mask", "exclusion mask", "composite"])

      px_filled = calc_fill_percent(compositeMask, totalArea)

      categName = self.ann_category_name(chosenAnn)
      attributes['category_percentage'][categName] += px_filled-prev_px_filled
      
      # include which objects are filled
      attributes['text_metadata']['objects'].append(categName)
      attributes['text_metadata']['descriptions'].append(randCoco['caption'])
      # print_generator_status(attributes, px_filled, skipped)
      pbar.update(px_filled-prev_px_filled)
      prev_px_filled = px_filled
    pbar.close()

    composite = Image.fromarray(composite)
    return composite, attributes  

  # generates image, metadata, and descriptions
  def generate_botr(self, config, outpath=None):
    image, metadata = self.generate_image(config, imageProgress=False)
    _, description = zipf_description(metadata, sentence_len=random.randint(3, 14))
    name = generate_name(metadata["category_percentage"])
    metadata["name"] = name
    metadata["description"] = description
    if outpath is not None:
      save_asset_metadata_pair(outpath, image, metadata)
    return image, metadata