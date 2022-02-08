
from copy import copy
import enum

import plotly.graph_objects as go
import random
import cv2
from skimage.exposure import histogram
import math

from torch import normal

from dataset import composition_attributes, get_annotation_supercategory
import numpy as np
from skimage.exposure import match_histograms, cumulative_distribution, equalize_adapthist
from skimage import img_as_float
from metaplex import generate_metadata
from pyramids import blend_masked_rgb
from masking import resize_fit, create_exclusion_mask, mask_add_composite, calc_fill_percent, add_images, mask_image
from utils import print_pretty, remove_file, save_object, save_asset_metadata_pair, find_nearest, sort_dict, imshow, load_json, load_object
from coco_utils import model_path, get_annotation_center
from language_model import LSTMTagger, generate_description_lstm
from language_processing import tokenize_sentence
from markov_language import Markov
from postprocessing import sharpen_image, jpeg_decimation, adaptive_hist, image_histogram
from PIL import Image, ImageOps
from tqdm import tqdm
from config import LSTM_CONFIG

from time import perf_counter


VOCAB_INFO = load_object(model_path("vocab_info"))
class BOTR_Layer():
    # add ability to move image components around
    # add ability to have this object determine which layers
    # get objects by area and relocate them
    def __init__(self, BOTR, coco_example, annotation=None):
        self.BOTR = BOTR
        self.coco_example = coco_example
        if annotation is None:
            annotation = coco_example.get_random_annotation(key=BOTR.config['ann_key'])
        self.annotation = annotation
        self.center = get_annotation_center(annotation)
        self.px_filled = 0

        self.raster = None

    # def recenter_annotation(self, center):
    #     current_ctr = get_annotation_center(self.annotation)

    def change_example(self, coco_example):
        # search the space for a certain kind of example
        # self.coco_example = BOTR.
        self.coco_example = coco_example
        

    # iterate through coco examples to find annotation with similar visual properties
    def find_similar_annotation(self, area_tolerance=0.01, pos_tolerance=0.01) -> bool:
        found_similar = False
        for i, example in enumerate(self.BOTR.Dataset):
            if example == self.coco_example or len(example.areas_all) == 0:
                continue
            current_area = self.coco_example.get_annotation_area(self.annotation)
            new_ann, area = example.closest_ann_area(current_area)
            areaDiff = abs(area - current_area)
            if areaDiff < area_tolerance:
                new_ctr = get_annotation_center(new_ann)
                dist = math.dist(new_ctr, self.center)
                if dist < pos_tolerance:
                    self.coco_example = example
                    self.annotation = new_ann
                    self.center = new_ctr
                    found_similar = True
                    break
        return found_similar

    def shuffle_ann(self):
        self.annotation = self.coco_example.get_random_annotation()

    def update_raster(self, raster):
        self.raster = raster

        raster *= 255
        raster = np.clip(raster, 0, 255).astype(np.uint8)
        bw = np.asarray(ImageOps.grayscale(Image.fromarray(raster)))
        # print(f'BW {bw.dtype} {bw}')
        self.px_filled = np.count_nonzero(bw)

    def update_mask(self, compositeMask=None):
        if compositeMask is not None:
            mask = self.get_mask(compositeMask.shape)
            exclusionMask = create_exclusion_mask(mask, compositeMask)
            self.px_filled = np.count_nonzero(exclusionMask)
            return exclusionMask
        else:
            return self.BOTR.get_mask(self.annotation)

    def dominant_color(self, fit=[256,256]):
        img = self.mask_image(fit)
        hist = histogram(img, channel_axis=-1, normalize=True)[0]
        avg = np.mean(hist, axis=0)
        print(np.argmax(avg))

        return hist, img

    # ======== masking =================================

    # mask image without exclusion mask
    def mask_image(self, fit=None):
        return self.get_mask(fit) * self.get_image(fit)

    # raw mask layer without exclusion mask
    def get_mask(self, fit=None):
        mask = self.BOTR.get_mask(self.annotation)
        if fit is not None:
            mask = resize_fit(mask, [fit[0], fit[1]])
        return mask

    def render_exclusion_mask(self, compositeMask, config):
        mask = self.get_mask(compositeMask.shape)
        exclusionMask = create_exclusion_mask(mask, compositeMask)
        image = self.coco_example.load_image(fit=config['outputSize'])
        masked = image * exclusionMask
        self.px_filled = np.count_nonzero(masked)
        return masked, exclusionMask

    # ======== helpers =================================

    def get_image(self, fit=None):
        return self.coco_example.load_image(fit)

    def percentage_fill(self, imageSize):
        return self.px_filled/(imageSize[0]*imageSize[1])

# ************************************************************
# Class containing a list of botr layers, extending
# functionality of list
# ************************************************************
class Layers():

    def __init__(self):
        self.layers = []

    def __getitem__(self, idx):
        return self.layers[idx]
        
    def __len__(self):
        return len(self.layers)

    def append(self, layer: BOTR_Layer):
        self.layers.append(layer)

    def shuffle_order(self):
        random.shuffle(self.layers)

    def find_new_items(self, area_tolerance=0.01, pos_tolerance=0.01):
        print(f'=> finding similar annotations for {len(self.layers)} layers')
        found = 0
        for i, layer in enumerate(self.layers):
            if i % 10 == 0:
                print(f'finding new annotation for layer {i}/{len(self.layers)}')
            success = layer.find_similar_annotation(area_tolerance, pos_tolerance)
            if success:
                found += 1
        print(f'=> replaced {found} annotations')
    

    # remove n smallest images from canvas
    def remove_smallest_n(self, n: int=1):
        sizes = [l.px_filled for l in self.layers]
        remove_idx = np.argsort(sizes)[:n]
        for idx in remove_idx:
            self.layers.pop(idx)        

    # remove images from canvas under pixel threshold
    def clean_invisible(self, thresh: int=1):
        remove = [l for l in self.layers if l.px_filled < thresh]
        for layer in remove:
            self.layers.remove(layer)
        print(f'removed {len(remove)} layers from canvas')

# ************************************************************
# An object that contains rasterized generated assets created
# by a BOTR - works as a read-only state
# ************************************************************
class GeneratedItem():
    def __init__(self, image, config, metadata):
        self.image = image
        self.config = config
        self.metadata = metadata
    
    def display(self):
        imshow(self.image, self.metadata['text']['name'])
        print(f'Description: {self.metadata["text"]["description"]}')

    def save(self, outpath):
        save_asset_metadata_pair(
            outpath,
            self.image,
            self.metadata)

    def set_name(self, name: str) -> None:
        self.metadata['text']["name"] = name

    def set_description(self, description: str) -> None:
        self.metadata['text']["description"] = description

    def display_categories(self):
        display_items = [i for i in self.metadata["composition"].items() if i[1] > 0]
        labels = [i[0] for i in display_items]
        values = [i[1] for i in display_items]
        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(title = self.metadata['text']['name'])
        fig.show()

# ************************************************************
# An object that contains parameters and layers for
# generating a collage image
# ************************************************************
class BOTR():
  
    def __init__(self, config, dataset):
        # self.layers = []
        self.layers: Layers = Layers()
        self.config = config
        self.Dataset = dataset
        self.textMetadata = { "name" : "", "description" : "" }
        self.metadata = self.generate_metadata(config)
        # self.image = None
        self.history: GeneratedItem = [] # contains a generated botr and description
        self.generatedItem: GeneratedItem = None
        self.title_model = Markov()
        self.set_description_model(config['descriptionModel'])

    def display(self):
        self.generatedItem.display()

    def clear_layers(self):
        del self.layers
        self.layers = Layers()

    def generate_metadata(self, config=None) -> dict:
        if config is None:
            config = self.config
        metadata = {
            # nesting to preserve generated text upon new generations
            "text" : self.textMetadata,
            "composition" : self.get_composition_attributes(config) }
        return metadata

    # get metadata of the current composition
    def get_composition_attributes(self, 
            config: dict, normalize: bool=True) -> dict:
        attributes = composition_attributes()
        for layer in self.layers:
            categ = get_annotation_supercategory(layer.annotation)
            attributes[categ] += layer.percentage_fill(config['outputSize'])
        self.total_fill = sum(list(attributes.values()))
        if normalize and self.total_fill > 0:
            attributes = {k: v/self.total_fill for k, v in attributes.items()}
        return attributes
    
    # saves the image and metadata assets
    def save_assets(
           self, base_path: str, 
           genItem: GeneratedItem = None) -> None:

        self.save_state(genItem)
        if genItem:
            genItem.save(base_path)
        else:
            self.generatedItem.save(base_path)

    def set_description_model(self, model: str) -> None:
        if model == 'markov':
                self.descriptionModel = Markov()
        if model == 'lstm':
            self.descriptionModel = LSTMTagger(LSTM_CONFIG, 
                len(VOCAB_INFO['vocabulary']), len(VOCAB_INFO['all_tags']))

    def corpus(self) -> list:
        words = []
        for layer in self.layers:
            tokenized = tokenize_sentence(layer.coco_example.get_caption())
            words += tokenized
        return words

    # ======== generating ================================

    def save_state(self, generatedItem: GeneratedItem = None) -> None:
        if not generatedItem:
            generatedItem = copy(self.generatedItem)
        self.history.append(generatedItem)

    def generate_description(
            self, langParams: dict, 
            modelType: str = None, 
            genItem: GeneratedItem = None) -> str:

        # langParams={"seed" : "A", "iters" : 5, "length" : 10}
        if not modelType:
            self.set_description_model(modelType)
        if not genItem:
            # save state of existing item, prepare new one
            self.save_state(self.generatedItem)
            genItem = self.generatedItem

        corpus = self.corpus() if self.config['restrictCorpus'] else VOCAB_INFO['vocabulary']
        description = self.descriptionModel.generate_sentence(langParams, corpus)
        genItem.set_description(description)
        return description

    def generate_name(self, genItem: GeneratedItem = None, 
                        length: int=3) -> str:
        if not genItem:
            # save state of existing item, prepare new one
            self.save_state(self.generatedItem)
            genItem = self.generatedItem
        name = self.title_model.generate_word(
            {"length" : length},
            self.metadata['composition'].keys())
        genItem.set_name(name)
        return name

    # get the layer with an average histogram
    def average_layer_histogram(self, config, histSize=(256,256)) -> BOTR_Layer:
        histograms = {l: image_histogram(l.get_image(histSize), config) for l in self.layers}
        histograms = {k: v for k, v in histograms.items() if v is not None}
        if len(histograms) == 0: # give up, because sometimes histogram is wacky
            return self.layers[0].get_image()
        mean = np.mean(list(histograms.values()))
        histograms = {k: np.sum(np.abs(h-mean)) for k, h in histograms.items()}
        min_layer, _ = min(histograms.items(), key=lambda x: abs(mean - x[1]))
        return min_layer

    def render(self, config) -> Image:

        composite = np.zeros((config["outputSize"][0], config["outputSize"][1], 3), dtype=np.uint8)
        compositeMask = np.zeros((config["outputSize"][0], config["outputSize"][1], 1), dtype=np.float)
        # choose between automatic and manual color ref
        # consider using a reference layer instead, where properties are matched to layer
        if config["refLayerHistogram"] is not None:
            referenceImg = self.layers[config["refLayerHistogram"]].get_image(config["outputSize"])
        else:
            refLayer = self.average_layer_histogram(config)
            referenceImg = refLayer.get_image(config["outputSize"])

        pbar = tqdm(total=len(self.layers))

        for layer in self.layers:
            image = match_histograms(
                layer.get_image(config["outputSize"]), 
                referenceImg, 
                channel_axis=config['multichannelColorMatching'])

            if config['adaptiveHistogram']:
                image = adaptive_hist(image, config['adaptiveHistKernel'], config['adaptiveHistClip'])
            if config["jpeg_quality"] != 100:
                image = jpeg_decimation(image, config["jpeg_quality"], numpy=True)

            layerMasked, exclusionMask = layer.render_exclusion_mask(compositeMask, config)

            if config['compositeType'] == 'pyramid':
                composite = blend_masked_rgb(img_A=image, img_B=composite, 
                mask_A=exclusionMask, mask_B=compositeMask, blendConfig=config['image_blending'])
                # provide masked layer back to the layer
                layer.update_raster(layerMasked) # this is incorrect here, change eventually
            
            if config['compositeType'] == 'binary':
                composite = mask_add_composite(image, exclusionMask, composite)
                layer.update_raster(layerMasked)

            if config['compositeType'] == 'pro':
                kernel = np.ones((
                    config['proDilateKernel'],config['proDilateKernel']),np.uint8)
                exclusionMask = cv2.dilate(
                    exclusionMask.astype(np.float32), 
                    kernel, iterations = config['proDilateIter'])
                exclusionMask = cv2.GaussianBlur(
                    exclusionMask/256, (config['proKernel'], config['proKernel']), 0)
                exclusionMask = np.expand_dims(exclusionMask, -1)
                excluded = np.clip(image * exclusionMask, 0, 255)
                composite = np.clip(composite + excluded, 0, 255).astype(np.uint8)
                layer.update_raster(excluded)
            
            compositeMask = np.logical_or(exclusionMask, compositeMask).astype(np.uint8)
            pbar.update(1)

        composite = Image.fromarray(composite.astype(np.uint8))
        return composite

    def generate(self, config: dict = None) -> GeneratedItem:
        if not self.generator_ready():
            return None, None
        if not config:
            config = self.config
        if self.generatedItem is not None:
            self.save_state()
        generatedItem = GeneratedItem(
            self.render(config), config, self.generate_metadata(config))

        langParams = {"seed" : "A", "iters" : 5, "length" : random.randint(3, 15)}

        self.generate_description(
            langParams = langParams, genItem = generatedItem)
        self.generate_name(generatedItem, length=random.randint(2, 5))
        self.generatedItem = generatedItem
        return generatedItem

    # ======== imageops ================================

    def sharpen(self, iterations: int, 
            genItem: GeneratedItem = None) -> None:
        if not genItem:
            self.save_state(self.generatedItem)
            genItem = self.generatedItem
        genItem.image = sharpen_image(genItem.image, iterations)

    def jpeg_decimate(self, quality: int, 
            genItem: GeneratedItem = None) -> None:
        if not genItem:
            self.save_state(self.generatedItem)
            genItem = self.generatedItem
        genItem.image = jpeg_decimation(self.image, quality)

    # ======== helpers =================================

    def set_name(self, name: str) -> None:
        self.generatedItem.textMetadata["name"] = name

    def set_description(self, description: str) -> None:
        self.generatedItem.textMetadata["description"] = description

    def vocabulary(self) -> list:
        return list(set(self.corpus()))

    def generator_ready(self) -> bool:
        if len(self.layers) > 0:
            return True
        print(f'No active layers, unable to generate image')
        return False

    def check_update_config(self, config: dict = None) -> dict:
        if config is None:
            return self.config
        if config != self.config:
            self.set_description_model(config["descriptionModel"])
            self.config = config
            return self.config

    def get_mask(self, ann):
        return self.Dataset.get_mask(ann)

    def append_layer(self, layer : BOTR_Layer):
        self.layers.append(layer)

    # get the canvas fill in pixels or percent
    # render mode is more accurate but more time consuming
    def get_px_filled(self, percent: bool=True, render: bool=True) -> int:
        if render:
            comp = self.render(self.config).convert('L')
            fill = np.count_nonzero(comp)
        else:
            fill = sum([l.px_filled for l in self.layers])
        if percent:
            fill /= (self.config["outputSize"][0] * self.config["outputSize"][1])
        return fill
        
    # saves the object for future use
    def save_botr(self, path : str):
        del self.Dataset
        save_object(self, f"{path}/{self.metadata['name']}")

    def __getitem__(self, idx):
        return self.history[idx]

    def __len__(self):
        return len(self.history)

# instead of saving full descriptions to metadata, I should be saving coco links