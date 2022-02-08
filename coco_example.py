from torch import normal
from coco_utils import load_coco_image, get_annotation_center
# from dataset import get_annotation_supercategory
from utils import display_multiple_images, imshow, sort_dict
# from dataset import get_annotation_supercategory
import numpy as np
import random
import math

class COCO_Example():

    def __init__(self, data):
        self.data = data
        # sort the areas available
        self.areas_instances = self.sort_area("instance_ann")
        self.areas_stuff = self.sort_area("stuff_ann")
        self.areas_all = self.sort_area("any")

        # self.centers_all = self.get_annotation_centers("any")
        # self.selected_layer = None

    def load_image(self, fit=None, asarray=True, resize=None):
        image = load_coco_image(
            filename=self.data["filename"], 
            fit=fit, 
            asarray=asarray)
        if resize is not None:
            image = image.resize(resize)
        return image

    def display_original_image(self):
        imshow(self.load_image(), title=self.data['caption'])

    def sort_area(self, ann_key="any"):
        annList = self.get_annotations(ann_key)
        annAreas = {}
        for ann in annList: # normalize area
            annAreas[int(ann["id"])] = self.get_annotation_area(ann)
        annAreas = sort_dict(annAreas)
        return annAreas

    def closest_ann_area(self, area, ann_key="any"):
        areas = self.sort_area(ann_key)
        if len(areas) == 0:
            return None, None
        id, ann_area = min(areas.items(), key=lambda x: abs(area - x[1]))
        return self.annotation_by_id(id), ann_area

    def closest_ann_pos(self, posTarget: float=[], normalize: bool=True, ann_key: str="any"):
        centers = self.get_annotation_centers(ann_key, normalize)
        distances = {ann_id: math.dist(ctr, posTarget) for ann_id, ctr in centers.items()}
        closest = list(sort_dict(distances, reverse=False).items())
        ann = self.annotation_by_id(closest[0][0])
        return ann, closest[0][1]
        
    def annotation_by_id(self, id):
        annList = self.get_annotations("any")
        for ann in annList:
            if ann["id"] == id:
                return ann
        return None
        
    def get_annotations(self, key="any"):
        if key == "any": # just combine the two
            return self.data["stuff_ann"] + self.data["instance_ann"]
        else:
            return self.data[key]

    def get_random_annotation(self, key="any"):
        anns = self.get_annotations(key)
        if len(anns) > 0:
            return random.choice(anns) 
        return None
    
    # gets area relative to image size
    def get_annotation_area(self, ann):
        return ann['area'] / (self.data['dims'][0] *  self.data['dims'][1])

    # remove any annotations containing a supercategory
    # def remove_annotations(self, keys):
    #     for ann in self.get_annotation():
    #         categ = get_annotation_supercategory(ann)

    def get_annotation_centers(self, key="any", normalize=True):
        normDims = None
        if normalize:
            normDims = self.data['dims']
        anns = self.get_annotations(key)
        centers = {int(ann["id"]): get_annotation_center(ann, normDims) for ann in anns}
        return centers

    def get_caption(self):
        return self.data["caption"]

    def get_num_annotations(self):
        return len(self.areas_all)
