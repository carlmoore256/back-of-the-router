# from PIL.Image import composite, fromarray
from PIL import Image
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
from postprocessing import sharpen_image
from pyramids import blend_masked_rgb
from utils import load_dict, save_dict, filter_list, display_multiple_images, save_asset_metadata_pair
from coco_utils import load_coco_info, load_coco_image, generate_assets, coco_value_distribution, closest_sized_annotation, print_generator_status, load_coco_obj
from masking import resize_fit, create_exclusion_mask, mask_add_composite, calc_fill_percent
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
    _, description = generate_zipf_description(metadata, sentence_len=random.randint(3, 14))
    name = generate_name(metadata["category_percentage"])
    metadata["name"] = name
    metadata["description"] = description
    if outpath is not None:
      save_asset_metadata_pair(outpath, image, metadata)
    return image, metadata



def generate_zipf_description(metadata, sentence_len=10, plotDist=False):
    descriptions = metadata.copy().pop("text_metadata")['descriptions']

    all_words = []
    for d in descriptions:
        words = []
        for word in d.split():
            word = word.lower().replace(".", "")
            words.append(word)
            if "." in word:
                words.append("!")
        all_words += words

    zipf_chart = {k: 0 for k in list(set(all_words))}

    for word in all_words:
        zipf_chart[word] += 1
    zipf_chart = dict(sorted(zipf_chart.items(), key=lambda item: item[1], reverse=True))

    # generate distribution to pull from
    dist = np.random.exponential(scale=1, size=sentence_len)
    dist = ((dist - np.min(dist)) / (np.max(dist) - np.min(dist)) * (len(zipf_chart.keys())-1)).astype(int)

    if plotDist:
        plt.title("zipz chart word occurences")
        plt.hist(list(zipf_chart.values()), 50, density=True)
        plt.show()
        plt.title("exponential distribution")
        count, bins, ignored = plt.hist(dist, 50, density = True)
        plt.show()

    sorted_zipf = list(zipf_chart.keys())    
    sentence = [sorted_zipf[idx] for idx in dist]
    str_sentence = ''
    sentence_start = True
    for word in sentence:
        if sentence_start:
            str_sentence += f"{word[0].upper()}{word[1:]}" + " "
            sentence_start = False
        else:
            if word == ".":
                sentence_start = True
            str_sentence += word + " "
    return sentence, str_sentence

# def attribute_map(category_map)
# attributes = {cat["supercategory"]: 0 for cat in category_map.values()}

def generate_name(metadata):
  word_len = np.random.randint(3, 20)
  sorted_attrs = dict(sorted(metadata.items(), key=lambda item: item[1], reverse=True))
  name = ""
  attr_idx = 0
  while len(name) < word_len:
      key = list(sorted_attrs.keys())[attr_idx]
      slice_len = int((sorted_attrs[key] * word_len) ** 2)
      if slice_len == 0:
          slice_len = 1
      offset = random.randint(0, len(key)-slice_len-1)
      name += key[offset:offset+slice_len]
      attr_idx += 1
  return name

# ===== old stuff ===============================


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

    composite = sharpen_image(composite.permute(2, 0, 1), sharpness).permute(1, 2, 0)

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
