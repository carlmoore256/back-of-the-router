
from matplotlib.pyplot import hist
from dataset import create_attribute_dict, get_annotation_supercategory
import numpy as np
from skimage.exposure import match_histograms, histogram, cumulative_distribution, equalize_adapthist
from skimage import img_as_float
from pyramids import blend_masked_rgb
from masking import resize_fit, create_exclusion_mask, mask_add_composite, calc_fill_percent, add_images, mask_image
from utils import print_pretty, save_object, save_asset_metadata_pair, find_nearest, sort_dict, imshow, load_json, load_object
from coco_utils import model_path
from language_model import LSTMTagger, generate_description_lstm
from language_processing import generate_name, tokenize_sentence
from markov_language import Markov
from postprocessing import sharpen_image, jpeg_decimation, adaptive_hist
from PIL import Image
from tqdm import tqdm
from config import LSTM_CONFIG

VOCAB_INFO = load_object(model_path("vocab_info"))
class BOTR_Layer():

    def __init__(
        self, BOTR, coco_example, annotation):
        # self.Dataset = dataset
        self.BOTR = BOTR
        self.coco_example = coco_example
        self.annotation = annotation
        self.center = coco_example.get_annotation_center(annotation)
        # self.maskedImage = mask_image(image, exclusionMask)
        self.px_filled = np.count_nonzero(exclusionMask)


    def update_mask(self, compositeMask=None):
        if compositeMask is not None:
            mask = resize_fit(self.BOTR.get_mask(self.annotation),
             [compositeMask.shape[0], compositeMask.shape[1]])
            exclusionMask = create_exclusion_mask(mask, compositeMask)
            self.px_filled = np.count_nonzero(exclusionMask)
            return exclusionMask
        else:
            return self.BOTR.get_mask(self.annotation)

    def get_image(self, fit=None):
        return self.coco_example.load_image(fit)

    def dominant_color(self):
        hist = histogram(self.mask_image())[0]
        avg = np.mean(hist, axis=0)
        print(np.argmax(avg))


    # mask image without exclusion mask
    def mask_image(self, fit=None):
        return self.get_mask(fit) * self.get_image(fit)

    # raw mask layer without exclusion mask
    def get_mask(self, fit=None):
        mask = self.BOTR.get_mask(self.annotation)
        if fit is not None:
            mask = resize_fit(mask, [fit[0], fit[1]])
        return mask

    def render(self, compositeMask, config):
        # # we don't need to remask
        # if compositeMask == self.compositeMask:
        #     return self.maskedImage
        # resize mask to fit
        # mask = resize_fit(
        #     self.BOTR.get_mask(self.annotation), 
        #     [compositeMask.shape[0], compositeMask.shape[1]])
        mask = self.get_mask(compositeMask.shape)
        # generate new exclusion mask
        exclusionMask = create_exclusion_mask(mask, compositeMask)
        image = self.coco_example.load_image(fit=config['outputSize'])
        return mask_image(image, exclusionMask)

# an object that contains a generated botr
class BOTR():
  
    def __init__(self, config, dataset):
        self.layers = []
        self.config = config
        self.Dataset = dataset
        self.metadata = None
        # create_attribute_dict()
        self.image = None
        self.composites = []
        self.set_description_model(config)

    def append_layer(self, layer : BOTR_Layer):
        self.layers.append( layer )
        # self.layers.append(layer)

    def update_metadata(self, config):
        if self.metadata is None:
            self.metadata = self.generate_metadata(config)
            return
        for category in list(self.metadata['category_percentage'].keys()):
            self.metadata['category_percentage'][category] = 0
        for layer in self.layers:
            category = get_annotation_supercategory(layer.annotation)
            self.metadata['category_percentage'][category] += layer.px_filled / (config["outputSize"][0]*config["outputSize"][1])

    def generate_metadata(self, config):
        if config is None:
            config = self.config
        metadata = create_attribute_dict()

        for layer in self.layers:
            category = get_annotation_supercategory(layer.annotation)
            metadata['category_percentage'][category] += layer.px_filled / (config["outputSize"][0]*config["outputSize"][1])
            metadata['text_metadata']['objects'].append(category)
            metadata['text_metadata']['descriptions'].append(layer.coco_example.get_caption())
        return metadata

    # this is fucking trash because it took hours to try to figure out
    # why histogram doesn't consistenly produce arrays of the same shape
    # hist size exists to load the image as a lightweight small version
    # for comparing histograms, and is not used during image compilation
    def average_layer_histogram(self, config, histSize=(256,256)):
        images = [l.get_image(histSize) for l in self.layers]
        histograms = {}
        for i, img in enumerate(images):
            try:
                hist = histogram(img, 
                        normalize=config['normalizeHistogram'],
                        channel_axis=config['histogramChannelAxis'])[0]
                if hist.shape[-1] == histSize[0]:
                    histograms[i] = hist
            except Exception as e:
                print(e)
        if len(list(histograms.items())) == 0: # literally just give up theres no easy way
            return images[0]
        mean = np.median(list(histograms.values()))
        # diffs = [np.sum(np.abs(h - mean)) for h in histograms.values()]

        for k, h in histograms.items():
            histograms[k] = np.sum(np.abs(h - mean))

        sorted_h = sort_dict(histograms, reverse=False)
        # print(list(sorted_h.keys()))
        return images[list(sorted_h.keys())[0]]

        # print(f'SORTED H {sorted_h}')
        # idx_min = np.argmin(diffs)
        # return images[idx_min] #FIX EVENTUALLY


    def corpus(self):
        words = []
        for layer in self.layers:
            words += tokenize_sentence(layer.coco_example.get_caption())
        return words

    def vocabulary(self):
        return list(set(self.corpus()))

    def get_mask(self, ann):
        return self.Dataset.get_mask(ann)

    def set_current_image(self, index : int):
        self.image = self.composites[index]

    def get_current_image(self):
        return self.image

    def display_current_image(self):
        imshow(self.image)

    def sharpen(self, iterations : int):
        self.image = sharpen_image(self.image, iterations)
        self.composites.append(self.image)

    def jpeg_decimate(self, quality : int):
        self.image = jpeg_decimation(self.image, quality)
        self.composites.append(self.image)

    # saves the object for future use
    def save_botr(self, path : str):
        # del self.Dataset # this can be loaded in during runtime
        del self.Dataset
        save_object(self, f"{path}/{self.metadata['name']}")
    
    # saves the image and metadata assets
    def save_assets(self, outpath : str, image_index : int = None):
        if image_index is not None:
            save_asset_metadata_pair(outpath, self.composites[image_index], self.metadata)
        else:
            save_asset_metadata_pair(outpath, self.image, self.metadata)

    def set_name(self, name : str):
        self.metadata["name"] = name

    def set_description(self, description : str):
        self.metadata["description"] = description

    def set_description_model(self, config):
        if config['restrictCorpus']:
            config['corpus'] = self.corpus()
            vocabSize = len(self.vocabulary())
            tagsetSize = len(VOCAB_INFO['all_tags'])
        else:
            config['corpus'] = self.Dataset.text_corpus()
            vocabSize = len(VOCAB_INFO['vocabulary'])
            tagsetSize = len(VOCAB_INFO['all_tags'])

        if config['descriptionModel'] == 'markov':
                self.descriptionModel = Markov(config['corpus'])
        if config['descriptionModel'] == 'lstm':
            self.descriptionModel = LSTMTagger(LSTM_CONFIG, 
                vocabSize, tagsetSize)
        

    def generate_description(self, config, modelType=None):
        if modelType is not None:
            self.set_description_model(modelType)
        # config={"seed" : "A", "iters" : 5, "corpus" : None}
        self.metadata["description"] = self.descriptionModel.generate(config)
        return self.metadata["description"]

    def generate_name_description(self):
        self.set_description(generate_description_lstm(self.metadata))
        self.set_name(generate_name(self.metadata))

    def generator_ready(self):
        if len(self.layers) > 0:
            return True
        print(f'No active layers, unable to generate image')
        return False

    def check_update_config(self, config):
        if config is None:
            return self.config
        if config != self.config:
            self.set_description_model(config)
            self.config = config
            return self.config

    def render(self, config=None):
        composite = np.zeros((config["outputSize"][0], config["outputSize"][1],3), dtype=np.uint8)
        compositeMask = np.zeros((config["outputSize"][0], config["outputSize"][1], 1), dtype=np.float)
        # choose between automatic and manual color ref
        if config["refLayerHistogram"] is not None:
            referenceImg = self.layers[config["refLayerHistogram"]].get_image(config["outputSize"])
        else:
            referenceImg = self.average_layer_histogram(config)
         
        pbar = tqdm(total=len(self.layers))

        for layer in self.layers:
            layer.update_mask(compositeMask)
            image = match_histograms(
                layer.get_image(config["outputSize"]), 
                referenceImg, 
                channel_axis=config['multichannelColorMatching'])

            if config['adaptiveHistogram']:
                image = adaptive_hist(image, config['adaptiveHistKernel'], config['adaptiveHistClip'])
            if config["jpeg_quality"] != 100:
                image = jpeg_decimation(image, config["jpeg_quality"], numpy=True)
           
            if config['image_blending']['use_blending']:
                composite = blend_masked_rgb(img_A=image,img_B=composite, 
                mask_A=layer.exclusionMask, mask_B=compositeMask, blendConfig=config['image_blending'])
            else:
                composite = mask_add_composite(image, layer.exclusionMask, composite)

            compositeMask = np.logical_or(
                layer.exclusionMask, compositeMask).astype(np.uint8)
            pbar.update(1)
        composite = Image.fromarray(composite.astype(np.uint8))
        self.composites.append(composite)
        return composite

    def generate(self, config=None):
        if not self.generator_ready():
            return None, None
        self.check_update_config(config)
        self.image = self.render(config)
        self.update_metadata(config)
        if "name" not in self.metadata.keys() or "description" not in self.metadata.keys():
            self.generate_name_description()
        return self.image, self.metadata



# def average_layer_histogram(self, config):
#     images = [l.get_image(config["outputSize"]) for l in self.layers]
#     histograms = {}
#     for i, img in enumerate(images):
#         try:
#             hist = histogram(img, 
#                     normalize=config['normalizeHistogram'],
#                     channel_axis=config['histogramChannelAxis'])[0]
#             if hist.shape[-1] == config["outputSize"][0]:
#                 histograms[i] = hist
#         except Exception as e:
#             print(e)
#     if len(list(histograms.items())) == 0: # literally just give up theres no easy way
#         return images[0]
#     mean = np.mean(list(histograms.values()))
#     # diffs = [np.sum(np.abs(h - mean)) for h in histograms.values()]

#     for k, h in histograms.items():
#         histograms[k] = np.sum(np.abs(h - mean))

#     sorted_h = sort_dict(histograms, reverse=False)
#     print(list(sorted_h.keys()))
#     return images[list(sorted_h.keys())[0]]