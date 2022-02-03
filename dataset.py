# contains functions for downloading the coco dataset
from utils import download_file, unzip_file, load_object, save_object, remove_file, check_make_path, print_pretty
from coco_utils import generate_assets, check_missing_assets, coco_value_distribution, load_asset, load_coco_obj, model_path
from language_processing import tokenize_sentence, get_all_possible_tags
from coco_example import COCO_Example
from config import DATASET_CONFIG

import numpy as np
import argparse
import random
import os

# this can be global
COCO_CATEGORIES = load_object(DATASET_CONFIG["category_map"])

class Dataset():

    def __init__(self, subset="coco-safe-licenses"):
        # checks for all necessary dataset files
        check_missing_assets()
        # load these as previously saved objects
        self.coco_examples = load_asset("saved-objects", subset)
        for k, v in self.coco_examples.items():
            # replace the value with a COCO_Example object
            self.coco_examples[k] = COCO_Example(v)
        self.cocoCaptions = load_coco_obj("captions")
        self.all_ids = list(self.coco_examples.keys())
        self.distStuff, self.meanAreaStuff, self.stdAreaStuff = coco_value_distribution(
        self.coco_examples, key="stuff_ann")
        self.distInstances, self.meanAreaInstances, self.stdAreaInstances = coco_value_distribution(
        self.coco_examples, key="instance_ann")

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

    # get all words for all descriptions, tokenized
    def text_corpus(self):
        sentences = [example.get_caption() for _, example in enumerate(self)]
        corpus = []
        for s in sentences:
            corpus += tokenize_sentence(s)
        return corpus

    # get a set of all word occurences
    def vocabulary(self):
        return list(set(self.text_corpus()))

    def get_vocab_info(self):
        vocab_info = load_object(model_path("vocab_info"))
        if vocab_info is None:
            vocab_info = self.save_vocab_info()
        return vocab_info

    def save_vocab_info(self):
        vocab = self.vocabulary()
        all_tags, tag_to_id, id_to_tag = get_all_possible_tags(vocab)
        word_to_id = {w: idx for (idx, w) in enumerate(vocab)}
        id_to_word = {idx: w for (idx, w) in enumerate(vocab)}
        vocab_info = {
            "all_tags" : all_tags,
            "tag_to_id" : tag_to_id,
            "id_to_tag" : id_to_tag,
            "vocabulary" : vocab,
            "word_to_id" : word_to_id,
            "id_to_word" : id_to_word }
        save_object(vocab_info, model_path("vocab_info"))
        return vocab_info

    # iterable functionality
    def __getitem__(self, idx):
        return self.coco_examples[self.all_ids[idx]]

    def __len__(self):
        return len(self.coco_examples)

# ===========================================================================

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