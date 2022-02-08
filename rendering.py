# rendering functions for botr

import random

from botr import BOTR, BOTR_Layer
from dataset import Dataset, get_annotation_supercategory

SEARCH_OPTS = {
    "areaTarget" : 0.03,
    "areaTolerance" : 0.001,
    "posTarget" : [0.5, 0.5], # normalized target
    "posTolerance" : 100,
    "ann_type" : "any",
    "exclusions" : ["person", "vehicle"]
}

def layer_target_area(dataset: Dataset, areaTarget: float=0.03, 
                        tolerance: float = 0.001) -> BOTR_Layer:
    return dataset.candidates_target_area(areaTarget)

def is_allowed_category(ann, exclusions):
    if get_annotation_supercategory(ann) in exclusions:
        return False
    return True

# fill with random sized patches
# def fill_target_area(botrGen: BOTR, areaTarget: float=0.03, 
#                         jitter: float=0.1, tolerance: float = 0.001,
#                         layers: int=100, exclusions: str=[]):
    
#     while len(botrGen.layers) < layers:
#         example, ann = botrGen.Dataset.example_target_area(
#             botrGen.Dataset, 
#             areaTarget * ((jitter * random.random()) + 1),
#             tolerance)
#         if is_allowed_category(ann, exclusions):
#             botrGen.append_layer(BOTR_Layer(botrGen, example, ann))

# fill with random sized patches
def fill_target_area(botrGen: BOTR, searchOpts: dict, 
                        jitter: float=0.1, layers: int=100):

    candidates = botrGen.Dataset.candidates_target_area(
        areaTarget=searchOpts['areaTarget'] * ((jitter * random.random()) + 1),
        areaTolerance=searchOpts['areaTolerance'],
        ann_type=searchOpts['ann_type'])
    candidates = list(candidates.items())
    random.shuffle(candidates)

    while len(botrGen.layers) < layers:
        example, ann = candidates.pop()
        if is_allowed_category(ann, searchOpts['exclusions']):
            botrGen.append_layer(BOTR_Layer(botrGen, example, ann))


def add_single_patch(botrGen: BOTR, searchOpts: dict, jitter: float=0.1):
    example, ann = examples_search_opts(botrGen.Dataset, searchOpts)[0]
    botrGen.append_layer(BOTR_Layer(botrGen, example, ann))

def add_multi_patch(botrGen: BOTR, searchOpts: dict, numPatches: int=10):
    botrCurrentLayers = len(botrGen.layers)

    matches = examples_search_opts(botrGen.Dataset, searchOpts)
    random.shuffle(matches)

    while len(botrGen.layers) < botrCurrentLayers + numPatches:
        if len(matches) > 1:
            example, ann = matches.pop()
        else:
            matches = examples_search_opts(botrGen.Dataset, searchOpts)
            random.shuffle(matches)
            example, ann = matches.pop()
        botrGen.append_layer(BOTR_Layer(botrGen, example, ann))

# gets an example with annotation using a dict of search options
def examples_search_opts(dataset: Dataset, searchOpts: dict):
    
    areas = dataset.candidates_target_area(
        areaTarget=searchOpts['areaTarget'],
        areaTolerance=searchOpts['areaTolerance'],
        ann_type=searchOpts['ann_type'])

    positions = dataset.candidates_target_pos(
        posTarget=searchOpts['posTarget'],
        posTolerance=searchOpts['posTolerance'],
        ann_type=searchOpts['ann_type'])

    matches = list(areas.keys() & positions.keys())
    
    # rand_example = random.choice(matches)
    # rand_ann = positions[rand_example]

    return [[match, positions[match]] for match in matches]


# def combine_two(data1, data2):
