from utils import load_coco_info
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import random
import os
from PIL import Image, ImageOps
import pickle

def sort_coco(coco_stuff, coco_instances, coco_captions):
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

def save_sorted(coco_images, path):
    with open(path, 'wb') as handle:
        pickle.dump(coco_images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
def get_available_images(annotations):
    return sorted(list(set(([key['image_id'] for key in annotations]))))

if __name__ == "__main__":
    coco_instances = COCO("dataset/annotations/instances_train2017.json")
    coco_stuff = COCO("dataset/annotations/stuff_train2017.json")
    coco_captions = load_coco_info("dataset/annotations/captions_train2017.json")

    avail_caption_imgs = get_available_images(coco_captions['annotations'])
    avail_instance_imgs = get_available_images(coco_instances.dataset['annotations'])
    avail_stuff_imgs = get_available_images(coco_stuff.dataset['annotations'])

    sorted_coco = sort_coco(coco_stuff, coco_instances, coco_captions)

    save_sorted(sorted_coco, path='dataset/coco_organized.pickle')