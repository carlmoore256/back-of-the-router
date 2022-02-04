# from utils import load_coco_info
from utils import save_object, load_object, load_json, save_json, filter_dict_by
from config import DATASET_CONFIG, SUPERCATEGORIES
from pycocotools.coco import COCO
from PIL import Image, ImageOps
import numpy as np
import pickle
import json
import os

def get_annotation_center(annotation):
    box = annotation["bbox"]
    l_top_x = box[0]
    l_top_y = box[1]
    width = box[2]
    height = box[3]
    return [l_top_x + (width/2), l_top_y - (height/2)]

def all_category_names(exclude=[]):
    names = SUPERCATEGORIES.copy()
    for e in exclude:
        names.remove(e)
    return names

def category_map(coco_stuff, coco_instances):
    category_map = {}
    for cat in coco_stuff.dataset['categories']:
        category_map[cat['id']] = cat
    for cat in coco_instances.dataset['categories']:
        category_map[cat['id']] = cat
    category_map = dict(
        sorted(category_map.items(), key=lambda item: item[1]["id"], reverse=False))
    return category_map

def sort_coco(coco_stuff, coco_instances, coco_captions):
    avail_caption_imgs = get_available_images(coco_captions['annotations'])
    avail_instance_imgs = get_available_images(coco_instances.dataset['annotations'])
    avail_stuff_imgs = get_available_images(coco_stuff.dataset['annotations'])

    valid_imgs = sorted(set.intersection(*map(set,[avail_caption_imgs, avail_instance_imgs, avail_stuff_imgs])))
    coco_images = {}
    sorted_stuff = {}

    for ann in coco_stuff.dataset['annotations']:
        img_id = ann['image_id']
        if img_id in sorted_stuff.keys():
            sorted_stuff[img_id].append(ann)
        else:
            sorted_stuff[img_id] = []

    sorted_instances = {}

    for instance in coco_instances.dataset['annotations']:
        img_id = instance["image_id"]
        if instance["image_id"] in sorted_instances.keys():
            sorted_instances[img_id].append(instance)
        else:
            sorted_instances[img_id] = []

    sorted_captions = {}
    for cap in coco_captions['annotations']:
        sorted_captions[cap['image_id']] = cap

    licenses = {}
    for l in coco_captions['licenses']:
        licenses[l['id']] = l

    sorted_images = {}
    for img in coco_captions['images']:
        sorted_images[img['id']] = img

    for img_id in valid_imgs:
        image = sorted_images[img_id]

        annotation = sorted_captions[img_id]

        instance_ann = sorted_instances[img_id]

        stuff_ann = sorted_stuff[img_id]

        coco_images[img_id] = {
            "filename" : image['file_name'],
            "dims" : (image['height'], image['width']),
            "date" : image["date_captured"],
            "license_id" : int(licenses[image['license']]['id']),
            "license_name" : licenses[image['license']]['name'],
            "ann_id" : image['id'],
            "caption" : annotation["caption"],
            "caption_id" : annotation["id"],
            "instance_ann" : instance_ann,
            "stuff_ann" : stuff_ann,
        }
    return coco_images

    
def load_coco_info(path="annotations/stuff_val2017.json"):
    with open(path,'r') as COCO:
        info = json.loads(COCO.read())
    return info

def load_coco_categories(path="annotations/stuff_val2017.json"):
    with open(path,'r') as COCO:
        categories = json.loads(COCO.read())['categories']
    categories_id = {}
    for cat in categories:
        categories_id[cat['id']] = cat
    return categories_id

def load_coco_image(filename, fit=None, asarray=True):
    path = os.path.join(
        DATASET_CONFIG["base_path"], 
        DATASET_CONFIG["images"], 
        filename)
    img = Image.open(path)
    if fit is not None:
        img = ImageOps.fit(img, fit).convert('RGB')
    if img.mode != "RGB":
        img = img.convert('RGB')
    if asarray:
        img = np.asarray(img)
    return img

def load_coco_obj(asset_name):
    filepath = asset_path("annotations", asset_name)
    print(f'=> loading coco object - {filepath}')
    return COCO(filepath)

def get_available_images(annotations):
    return sorted(list(set(([key['image_id'] for key in annotations]))))

# filter collection (default is to a set of allowed licenses)
def filter_sorted_coco(sorted_coco, key='license_id', allowed=[2,5,7]):
    filtered_coco = dict(filter(lambda elem: elem[1][key] in allowed, sorted_coco.items()))
    return filtered_coco

# given a list of annotations, return the one closest in size 
def closest_sized_annotation(annotationList, targetSize):
    return min(annotationList, key=lambda x: abs(targetSize - x['area']))

# calculate global area distribution for fine grained control of patch sizes
def coco_value_distribution(coco_dataset, key="stuff_ann"):
    dist = []
    for coco in coco_dataset.values():
        vals = list(coco.sort_area(key).values())
        dist += vals
    return dist, np.mean(dist), np.std(dist)

def print_generator_status(attributes, percentFill, skipped):
    print(f'filled {percentFill}% skipped {skipped}')

def asset_path(asset_categ : str, asset_name : str):
    if asset_name not in DATASET_CONFIG["assets"][asset_categ].keys():
        DATASET_CONFIG["assets"][asset_categ][asset_name] = f'{asset_name}.pickle'
    return os.path.join(DATASET_CONFIG["base_path"], DATASET_CONFIG["assets"][asset_categ][asset_name])

# where model related things are saved
def model_path(key : str):
    return os.path.join(DATASET_CONFIG['base_path'], DATASET_CONFIG['model'][key])

def load_asset(asset_categ : str, asset_name : str):
    filepath = asset_path(asset_categ, asset_name)
    if not os.path.isfile(filepath):
        print(f'[!] {filepath} not found, generating asset!')
        generate_assets()
    print(f'=> loading coco asset: {filepath}')
    return load_object(filepath)

def check_missing_assets():
    for categ_key, categ_items in DATASET_CONFIG["assets"].items():
        for name_key, _ in categ_items.items():
            filepath = asset_path(categ_key, name_key)
            if not os.path.isfile(filepath):
                print(f'[!] {filepath} not found, generating asset!')
                generate_assets()
                break

def generate_assets():
    coco_instances = COCO(asset_path("annotations", "instances"))
    coco_stuff = COCO(asset_path("annotations", "stuff"))
    coco_captions = load_coco_info(asset_path("annotations", "captions"))

    sorted_coco = sort_coco(coco_stuff, coco_instances, coco_captions)

    for filter, values in DATASET_CONFIG["filters"].items():
        print(f'=> filtering coco by {filter}, allowed values: {values}')
        filtered_coco = filter_dict_by(sorted_coco, filter, values)

    # saved a sorted and filtered set of coco objects
    save_object(filtered_coco, asset_path("saved-objects", "coco-safe-licenses"))
    
    # save the mapping of categories
    if not os.path.isfile(DATASET_CONFIG["category_map"]):
        cat_map = category_map(coco_stuff, coco_instances) 
        save_object(cat_map, DATASET_CONFIG["category_map"])

# sort coco dataset in desired BOTR format, serialize and save
if __name__ == "__main__":
    generate_assets()