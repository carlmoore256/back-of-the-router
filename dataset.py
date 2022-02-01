# contains functions for downloading the coco dataset
from utils import download_file, unzip_file, load_json, remove_file, check_make_path, print_pretty
from coco_utils import generate_assets, check_missing_assets, coco_value_distribution, load_asset, load_coco_obj, closest_sized_annotation
from config import DATASET_CONFIG

import numpy as np
import argparse
import random
import os

# this can be global
COCO_CATEGORIES = load_asset("saved-objects", "category_map")

class Dataset():

    def __init__(self):
        # checks for all necessary dataset files
        check_missing_assets()

        # load these as previously saved objects
        self.coco_examples = load_asset("saved-objects", "coco_organized")
        self.cocoCaptions = load_coco_obj("captions")

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

    # extends functionality to retrive both stuff and instances
    # def get_random_ann_list(self, ann_key):
    #     randCoco = self.get_coco_example()
    #     if ann_key == "any":
    #         annList = randCoco["stuff_ann"]
    #         annList += randCoco["instance_ann"]
    #         return annList
    #     else:
    #         return randCoco[ann_key]


# generates an empty attribute dict
def create_attribute_dict():
    attributes = {
        "category_percentage": 
        {   cat["supercategory"]: 0 for cat in COCO_CATEGORIES.values() }}
    attributes['text_metadata'] = {
        "objects" : [], "descriptions" : [] }
    return attributes

def get_annotation_supercategory(annotation):
    name = COCO_CATEGORIES[annotation["category_id"]]["supercategory"]
    return name

def filter_annotation_categories(annotations, allowedCategories):
    # map string category names to annotations
    annCategories = {get_annotation_supercategory(ann) : ann for ann in annotations}
    # construct list of allowed values
    filtered = [v for k, v in annCategories.items() if k in allowedCategories]
    return filtered

def download_dataset_archives(config):
    for filename, url in config["downloads"].items():
        print(f"\n====== Retrieving {filename} ====")
        zip_path = os.path.join(config["temp_path"], filename)
        download_file(url, zip_path)
        unzip_file(zip_path, config["base_path"])
        remove_file(zip_path)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Initialize BOTR COCO Dataset')
    # parser.add_argument('--config', type=str, required=False, default="config/dataset_config.json", 
    #                     help='path to the dataset config json')
    # args = parser.parse_args()
    check_make_path(DATASET_CONFIG["base_path"])
    check_make_path(DATASET_CONFIG["temp_path"])
    download_dataset_archives(DATASET_CONFIG)
    generate_assets()