# contains functions for downloading the coco dataset
import math

from utils import download_file, unzip_file, load_object, save_object, remove_file, check_make_dir, print_pretty
from coco_utils import generate_assets, check_missing_assets, coco_value_distribution, load_asset, load_coco_obj, model_path, get_annotation_center
from language_processing import tokenize_sentence, get_all_possible_tags
from coco_example import COCO_Example
from config import DATASET_CONFIG

import numpy as np
import random
import os

# this can be global
COCO_CATEGORIES = load_object(DATASET_CONFIG["category_map"])

class Dataset():

    def __init__(self, subset="coco-safe-licenses"):
        # checks for all necessary dataset files
        check_missing_assets()
        # load these as previously saved objects
        self.cached_coco_examples = load_asset("saved-objects", subset)
        self.coco_examples = {}
        for k, v in self.cached_coco_examples.items():
            # replace the value with a COCO_Example object
            example = COCO_Example(v)
            if example.get_num_annotations() == 0:
                continue
            self.coco_examples[k] = example
        self.cocoCaptions = load_coco_obj("captions")
        self.all_ids = list(self.coco_examples.keys())
        self.distStuff, self.meanAreaStuff, self.stdAreaStuff = coco_value_distribution(
        self.coco_examples, key="stuff_ann")
        self.distInstances, self.meanAreaInstances, self.stdAreaInstances = coco_value_distribution(
        self.coco_examples, key="instance_ann")

    # get an example coco image, if no id is provided return a random one
    def get_coco_example(self, id: int=None) -> COCO_Example:
        if id is None:
            id = random.choice(self.all_ids)
        return self.coco_examples[id]

    # ===== retrieve filtered coco examples based on annotations ======

    # get a coco annotation and matching example wtih a matching type
    def example_ann_type(self, example, ann_type: str="any"):
        example = self.get_coco_example()
        ann = example.get_random_annotation(ann_type)
        if ann is not None:
            return example, ann
        return None, None

    # get a list of coco annotations and matching examples within an area constriant
    def candidates_target_area(self, example=None, area_target: float=0.05, 
                                area_tolerance: float=0.01, ann_type="any", 
                                sort=True, allowed_categ: list=None, limit: int=None):
        candidates = {}
        for _, [id, example] in enumerate(self.coco_examples.items()):
            ann, area = example.closest_ann_area(area_target, ann_type)
            if abs(area - area_target) < area_tolerance:
                # candidates.append([example, ann])
                if allowed_categ is None:
                    candidates[example] = ann
                elif get_annotation_supercategory(ann) in allowed_categ:
                    candidates[example] = ann
                if limit is not None and len(candidates) == limit:
                    break
        if len(candidates) > 0:
            if sort:
                candidates = {
                    k: v for k, v in 
                    sorted(candidates.items(), key=lambda item: item[0].get_annotation_area(item[1]))
                    }
            return candidates
        return None

    # get a list of coco annotations and matching examples within a position constraint
    def candidates_target_pos(self, example=None, pos_target: float=[],
                             pos_tolerance: float=0.1, ann_type: str="any"):
        candidates = {}
        for _, [id, example] in enumerate(self.coco_examples.items()):
            ann, dist = example.closest_ann_pos(pos_target, True, ann_type)
            if dist < pos_tolerance:
                # candidates.append([example, ann])
                candidates[example] = ann
        if len(candidates) > 0:
            return candidates
        return None, None

    def get_example_target_area(self, area_target, area_tolerance, ann_type):
        while True:
            example = self.get_coco_example()
            ann, area = example.closest_ann_area(area_target, ann_type)
            if abs(area - area_target) < area_tolerance:
                return example, ann

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
def composition_attributes():
    return { cat["supercategory"]: 0 for cat in COCO_CATEGORIES.values() }

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
    check_make_dir(DATASET_CONFIG["base_path"])
    check_make_dir(DATASET_CONFIG["temp_path"])
    download_dataset_archives(DATASET_CONFIG)
    generate_assets()