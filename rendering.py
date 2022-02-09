# rendering functions for botr

import random

from attr import attr

from botr import BOTR, BOTR_Layer
from dataset import Dataset, get_annotation_supercategory
from utils import image_nonzero_px

SEARCH_OPTS = {
    "areaTarget" : 0.03,
    "areaTolerance" : 0.001,
    "posTarget" : [0.5, 0.5], # normalized target
    "posTolerance" : 100,
    "ann_type" : "any",
    "exclusions" : ["person", "vehicle"]
}

QUICK_RENDER = {
    'matchHistograms' : False,
    'compositeType' : 'binary',
    'outputSize' : (64, 64),
    'jpeg_quality' : 100,
    'adaptiveHistogram' : False,
    'showProgress' : False
}

MEDIUM_RENDER = {
    'matchHistograms' : False,
    'compositeType' : 'binary',
    'outputSize' : (256, 256),
    'jpeg_quality' : 100,
    'adaptiveHistogram' : False,
    'showProgress' : False
}

def layer_target_area(dataset: Dataset, areaTarget: float=0.03, 
                        tolerance: float = 0.001) -> BOTR_Layer:
    return dataset.candidates_target_area(areaTarget)

def is_allowed_category(ann, exclusions):
    if get_annotation_supercategory(ann) in exclusions:
        return False
    else:
        return True

# fill with random sized patches
def fill_target_area_fast(botrGen: BOTR, searchOpts: dict,
                        jitter: float=0.1, layers: int=50):
    
    while len(botrGen.layers) < layers:
        example, ann = botrGen.Dataset.get_example_target_area(
            area_target = searchOpts['areaTarget'] * ((jitter * random.random()) + 1),
            area_tolerance = searchOpts['areaTolerance'],
            ann_type = searchOpts['ann_type'])
        if is_allowed_category(ann, searchOpts['exclusions']):
            botrGen.append_layer(BOTR_Layer(botrGen, example, ann))

# fill with random sized patches
def fill_target_area(botrGen: BOTR, searchOpts: dict, 
                        jitter: float=0.1, layers: int=50):

    candidates = botrGen.Dataset.candidates_target_area(
        area_target = searchOpts['areaTarget'] * ((jitter * random.random()) + 1),
        area_tolerance = searchOpts['areaTolerance'],
        ann_type = searchOpts['ann_type'])
    candidates = list(candidates.items())
    random.shuffle(candidates)

    while len(botrGen.layers) < layers:
        example, ann = candidates.pop()
        if is_allowed_category(ann, searchOpts['exclusions']):
            botrGen.append_layer(BOTR_Layer(botrGen, example, ann))


def add_single_patch(botrGen: BOTR, searchOpts: dict, jitter: float=0.1):
    example, ann = random.choice(examples_search_opts(botrGen.Dataset, searchOpts))
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
        area_target=searchOpts['areaTarget'],
        area_tolerance=searchOpts['areaTolerance'],
        ann_type=searchOpts['ann_type'])

    positions = dataset.candidates_target_pos(
        pos_target=searchOpts['posTarget'],
        pos_tolerance=searchOpts['posTolerance'],
        ann_type=searchOpts['ann_type'])

    matches = list(areas.keys() & positions.keys())
    
    # rand_example = random.choice(matches)
    # rand_ann = positions[rand_example]

    return [[match, positions[match]] for match in matches]

# fill a botr with a set of attributes to to target
def fill_target_attributes(botrGen: BOTR, attributes: dict, 
                                comp_target_fill: float=0.96,
                                overstep_limit: float=1.25) -> None:
    total_px = QUICK_RENDER['outputSize'][0]*QUICK_RENDER['outputSize'][1]
    for categ, target_fill in attributes.items():
        layer_fill = 0
        layers = []
        search_opts = {
            "areaTarget" : target_fill,
            "areaTolerance" : 0.001,
            "posTarget" : [0.5, 0.5],
            "posTolerance" : 1,
            "ann_type" : "any" 
        }

        prev_candidates = 0

        while layer_fill < target_fill:
            new_candidates = 0

            while prev_candidates == new_candidates:
                candidates = examples_search_opts(
                    botrGen.Dataset, search_opts)
                categories = [get_annotation_supercategory(ann) for _, ann in candidates]
                candidate_idxs = [i for i, cat in enumerate(categories) if cat == categ]
                candidates = [candidates[idx] for idx in candidate_idxs]
                new_candidates = len(candidates)
                search_opts["areaTolerance"] += 0.005

            prev_candidates = new_candidates
            print(f'found {len(candidates)} to place on canvas for {categ}')

            for ex, ann in candidates:
                layer = BOTR_Layer(botrGen, ex, ann)
                botrGen.append_layer(layer)
                layers.append(layer)
                
                # unfortunately as of now we have to render the whole image to
                # figure out how much fill each layer contributes
                # first, shuffle the order to achieve true randomness
                botrGen.layers.shuffle_order()
                rend = botrGen.render(QUICK_RENDER)
                total_fill = image_nonzero_px(rend) / total_px
                if total_fill > comp_target_fill:
                    return

                # iterate over layers we're adding, render to see what size is
                layer_fill = sum(
                    [l.percentage_fill() for l in layers])
                fill_ratio = layer_fill/target_fill

                print(f'Total: %{total_fill * 100} Layers: {len(botrGen.layers)} Filling {categ} - %{fill_ratio * 100}')
                if layer_fill > target_fill:
                    if (fill_ratio) > overstep_limit:
                        print(f'Overstepped %{(fill_ratio*100)-100} removing layer...')
                        botrGen.layers.remove(layer)
                        layers.pop(-1)
                    else:
                        break
                elif 1-fill_ratio < 0.1: # move it along
                    break

# fill a botr with a set of attributes to to target
def fill_target_attributes_fast(botrGen: BOTR, attributes: dict, 
                                comp_target_fill: float=0.96,
                                overstep_limit: float=1.25) -> None:
    total_px = QUICK_RENDER['outputSize'][0]*QUICK_RENDER['outputSize'][1]
    for categ, target_fill in attributes.items():
        layer_fill = 0
        layers = []
        search_opts = {
            "areaTarget" : target_fill,
            "areaTolerance" : 0.001,
            "posTarget" : [0.5, 0.5],
            "posTolerance" : 1,
            "ann_type" : "any"
        }

        while layer_fill < target_fill:

            example, ann = botrGen.Dataset.get_example_target_area(
                area_target = search_opts['areaTarget'],
                area_tolerance = search_opts['areaTolerance'],
                ann_type = search_opts['ann_type'])
            
            if get_annotation_supercategory(ann) != categ:
                continue

            layer = BOTR_Layer(botrGen, example, ann)
            botrGen.append_layer(layer)
            layers.append(layer)
            # shuffle order to get more optimal solution
            botrGen.layers.shuffle_order()
            rend = botrGen.render(QUICK_RENDER)
            total_fill = image_nonzero_px(rend) / total_px
            if total_fill > comp_target_fill:
                return

            # iterate over layers we're adding, render to see what size is
            layer_fill = sum(
                [l.percentage_fill() for l in layers])
            fill_ratio = layer_fill/target_fill

            print(f'Total: %{total_fill * 100} Layers: {len(botrGen.layers)} Filling {categ} - %{fill_ratio * 100}')
            if layer_fill > target_fill:
                if (fill_ratio) > overstep_limit:
                    print(f'Overstepped %{(fill_ratio*100)-100} removing layer...')
                    botrGen.layers.remove(layer)
                    layers.pop(-1)
                else:
                    break
            elif 1-fill_ratio < 0.1: # move it along
                break
    
# fill a botr until the percentage filled reaches target
def fill_to_target(botrGen: BOTR, search_opts: dict, 
                    target_fill=0.95, render_config: dict=MEDIUM_RENDER):

    total_px = render_config['outputSize'][0]*render_config['outputSize'][1]
    
    while True:
        example, ann = botrGen.Dataset.get_example_target_area(
                area_target = search_opts['areaTarget'],
                area_tolerance = search_opts['areaTolerance'],
                ann_type = search_opts['ann_type'])
        if ann in search_opts['exclusions']:
            continue

        layer = BOTR_Layer(botrGen, example, ann)
        botrGen.append_layer(layer)

        rend = botrGen.render(render_config)

        total_fill = image_nonzero_px(rend) / total_px
        if total_fill > target_fill:
            return


        if total_fill > 0.99:
            return
        image_nonzero_px(rend) / total_px