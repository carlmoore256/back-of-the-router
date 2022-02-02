from utils import display_multiple_images, save_asset_metadata_pair
from coco_utils import load_coco_image, closest_sized_annotation
from masking import resize_fit, create_exclusion_mask, mask_add_composite, calc_fill_percent, mask_image
from dataset import Dataset, filter_annotation_categories, get_annotation_supercategory, create_attribute_dict
from language_processing import generate_name, zipf_description
from postprocessing import sharpen_image
from pyramids import blend_masked_rgb
from config import GENERATOR_CONFIG_DEFAULT

from PIL import Image
from tqdm import tqdm
import numpy as np
import random


# a class that can remain loaded, and change aspects
# of generative properties
class BOTR_Generator():
  
  def __init__(self, subset="coco-safe-licenses"):
    self.Dataset = Dataset(subset)
  
  # select an annotation from a gaussian distribution
  def gaussian_select_patch(self, annList, annKey, avgSize=1., sizeVariance=1.):
    if len(annList) > 1:
      sizeTarget = self.Dataset.random_size_target(
        annKey, 
        sizeScalar=avgSize, 
        stdScalar=sizeVariance)
      return closest_sized_annotation(annList, sizeTarget)
    else:
      return annList[0]

  def generate_image(self, config, imageProgress=False, printWarnings=False):

    composite = np.zeros((config["outputSize"][0], config["outputSize"][1],3), dtype=np.uint8)
    compositeMask = np.zeros((config["outputSize"][0], config["outputSize"][1], 1), dtype=np.float)

    attributes = create_attribute_dict()
    totalArea = config['outputSize'][0]*config['outputSize'][1]

    px_filled = 0
    prev_px_filled = 0
    skipped = 0

    # components = [] # save each component of the constructed image along the way
    # botr = BOTR()

    pbar = tqdm(total=config['targetFill'])
    while px_filled < config['targetFill']:
      cocoExample = self.Dataset.get_coco_example()
      annList = cocoExample.get_annotation(config['ann_key'])

      # annList = self.get_random_ann_list(config['ann_key'])
      annList = filter_annotation_categories(annList, config["allowedCategories"])
      
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
      objectMask = self.Dataset.get_mask(chosenAnn)
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

      # image = load_coco_image(cocoExample['filename'], fit=config['outputSize'])
      image = cocoExample.load_image(fit=config['outputSize'])

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

      categName = get_annotation_supercategory(chosenAnn)
      attributes['category_percentage'][categName] += px_filled-prev_px_filled
      
      # include which objects are filled
      attributes['text_metadata']['objects'].append(categName)
      attributes['text_metadata']['descriptions'].append(cocoExample.get_caption())

      pbar.update(px_filled-prev_px_filled)
      prev_px_filled = px_filled
    pbar.close()

    composite = Image.fromarray(composite)
    return composite, attributes

  # def generate_botr_object(self, config=None, outpath=None):


  # generates image, metadata, and descriptions
  def generate_botr(self, config=None, outpath=None):
    if config is None:
      print(f'No config provided, using default config')
      config = GENERATOR_CONFIG_DEFAULT

    image, metadata = self.generate_image(config, imageProgress=False)
    _, description = zipf_description(metadata, sentence_len=random.randint(3, 14))
    name = generate_name(metadata["category_percentage"])
    metadata["name"] = name
    metadata["description"] = description
    if outpath is not None:
      save_asset_metadata_pair(outpath, image, metadata)
    return image, metadata

# an object that contains a generated botr
class BOTR():
  
  def __init__(self):
    self.layers = []
    return
  
  def append_layer(self, layer):
    self.layers.append(layer)
