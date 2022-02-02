from coco_utils import load_coco_image
# from dataset import get_annotation_supercategory
from utils import display_multiple_images, imshow, sort_dict
import numpy as np

class COCO_Example():

    def __init__(self, data):
        self.data = data
        # sort the areas available
        self.areas_instances = self.sort_area("instance_ann")
        self.areas_stuff = self.sort_area("stuff_ann")
        self.areas_all = self.sort_area("any")

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
        annList = self.get_annotation(ann_key)
        annAreas = {}
        for ann in annList:
            annAreas[int(ann["id"])] = ann['area']
        annAreas = sort_dict(annAreas)
        return annAreas

    def annotation_by_id(self, id):
        annList = self.get_annotation("any")
        for ann in annList:
            if ann["id"] == id:
                return ann
        return None
        
    def get_annotation(self, key="any"):
        if key == "any": # just combine the two
            return self.data["stuff_ann"] + self.data["instance_ann"]
        else:
            return self.data[key]

    def get_caption(self):
        return self.data["caption"]
