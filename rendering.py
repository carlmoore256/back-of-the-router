# rendering functions for botr

import random

from attr import attr
from construct import max_

from botr import BOTR, BOTR_Layer
from dataset import Dataset, get_annotation_supercategory
from utils import image_nonzero_px
from collections import OrderedDict
import numpy as np

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
    'showProgress' : False,
    "batchRenderSize" : 8
}

MEDIUM_RENDER = {
    'matchHistograms' : False,
    'compositeType' : 'binary',
    'outputSize' : (256, 256),
    'jpeg_quality' : 100,
    'adaptiveHistogram' : False,
    'showProgress' : False,
    "batchRenderSize" : 8
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
def fill_target_area_fast(botrGen: BOTR, dataset: Dataset, searchOpts: dict,
                        jitter: float=0.1, layers: int=50):
    
    while len(botrGen.layers) < layers:
        example, ann = dataset.get_example_target_area(
            area_target = searchOpts['areaTarget'] * ((jitter * random.random()) + 1),
            area_tolerance = searchOpts['areaTolerance'],
            ann_type = searchOpts['ann_type'])
        if is_allowed_category(ann, searchOpts['exclusions']):
            botrGen.append_layer(BOTR_Layer(example, ann))

# fill with random sized patches
def fill_target_area(botrGen: BOTR, dataset: Dataset, searchOpts: dict, 
                        jitter: float=0.1, layers: int=50):

    candidates = dataset.candidates_target_area(
        area_target = searchOpts['areaTarget'] * ((jitter * random.random()) + 1),
        area_tolerance = searchOpts['areaTolerance'],
        ann_type = searchOpts['ann_type'])
    candidates = list(candidates.items())
    random.shuffle(candidates)

    while len(botrGen.layers) < layers:
        example, ann = candidates.pop()
        if is_allowed_category(ann, searchOpts['exclusions']):
            botrGen.append_layer(BOTR_Layer(example, ann))


def add_single_patch(botrGen: BOTR, dataset: Dataset, 
                        searchOpts: dict, jitter: float=0.1):

    example, ann = random.choice(examples_search_opts(dataset, searchOpts))
    botrGen.append_layer(BOTR_Layer(example, ann))

def add_multi_patch(botrGen: BOTR, dataset: Dataset,
                        searchOpts: dict, numPatches: int=10):
    botrCurrentLayers = len(botrGen.layers)

    matches = examples_search_opts(dataset, searchOpts)
    random.shuffle(matches)

    while len(botrGen.layers) < botrCurrentLayers + numPatches:
        if len(matches) > 1:
            example, ann = matches.pop()
        else:
            matches = examples_search_opts(dataset, searchOpts)
            random.shuffle(matches)
            example, ann = matches.pop()
        botrGen.append_layer(BOTR_Layer(example, ann))

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
def fill_target_attributes(botrGen: BOTR, dataset: Dataset, attributes: dict, 
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
                    dataset, search_opts)
                categories = [get_annotation_supercategory(ann) for _, ann in candidates]
                candidate_idxs = [i for i, cat in enumerate(categories) if cat == categ]
                candidates = [candidates[idx] for idx in candidate_idxs]
                new_candidates = len(candidates)
                search_opts["areaTolerance"] += 0.005

            prev_candidates = new_candidates
            print(f'found {len(candidates)} to place on canvas for {categ}')

            for ex, ann in candidates:
                layer = BOTR_Layer(ex, ann)
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

def sum_candidate_pool(pool: list) -> float:
    return sum([c[0] for c in pool])

def sample_candidates(candidate_keys, candidates):
    example = random.sample(candidate_keys, 1)[0]
    ann = candidates[example]
    area = example.get_annotation_area(ann)
    return [area, {"example" : example, "ann" : ann}]

def sample_candidate_pool(candidate_keys, candidates, pass_target_fill):
    tolerance = 0.001
    tolerance_delta = 0.0001
    candidate_pool = [sample_candidates(candidate_keys, candidates)]
    total_area = sum([c[0] for c in candidate_pool])
    while True:
        [area, [example, ann]] = sample_candidates(candidate_keys, candidates)
        if abs(total_area + area - pass_target_fill) > tolerance:
            tolerance += tolerance_delta
            continue
        elif abs(total_area + area - pass_target_fill) < tolerance:
            print(f"candidate pool size: {total_area} len {len(candidate_pool)} target size: {pass_target_fill} diff {total_area/pass_target_fill}")
            return candidate_pool
        candidate_pool.append([area, [example, ann]])
        total_area = sum([c[0] for c in candidate_pool])


def fill_target_attributes_balanced(botrGen: BOTR, dataset: Dataset, attributes: dict, 
                                comp_target_fill: float=0.85,
                                overstep_lim: float=1.25,
                                iterations: int=5,
                                verbose=True) -> None:
    total_px = QUICK_RENDER['outputSize'][0]*QUICK_RENDER['outputSize'][1]
    ordered_attrs = OrderedDict(sorted(attributes.items(), key=lambda x:x[1], reverse=True))
    # area target should be average of areas
    attr_vals = np.asarray(list(ordered_attrs.values()))
    nonzero_attrs = attr_vals[attr_vals>0]
    mean_area = np.mean(nonzero_attrs)
    categ_candidates = {}
    for i in range(iterations):
        for categ, area_target in ordered_attrs.items():
            if area_target == 0:
                continue
            candidates = dataset.candidates_target_area(
                area_target = mean_area/area_target, 
                area_tolerance = 0.01, 
                ann_type="any", sort = True,
                allowed_categ=[categ],
                limit=10)
            

# fill a botr with a set of attributes to to target
def fill_target_attributes_fast(botrGen: BOTR, dataset: Dataset, attributes: dict, 
                                comp_target_fill: float=0.85,
                                overstep_lim: float=1.25,
                                verbose=True) -> None:
    total_px = QUICK_RENDER['outputSize'][0]*QUICK_RENDER['outputSize'][1]
    for categ, target_fill in attributes.items():
        # because the dict of categ will be ordered, we can increase the
        # tolerance as the categories become less significant
        overstep_lim += overstep_lim * 0.1 
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

            example, ann = dataset.get_example_target_area(
                area_target = search_opts['areaTarget'],
                area_tolerance = search_opts['areaTolerance'],
                ann_type = search_opts['ann_type'])
            
            if get_annotation_supercategory(ann) != categ:
                continue

            layer = BOTR_Layer(example, ann)
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

            if verbose:
                print(f'Total: %{total_fill * 100} Layers: {len(botrGen.layers)} Filling {categ} - %{fill_ratio * 100}')
            if layer_fill > target_fill:
                if (fill_ratio) > overstep_lim:
                    # print(f'Overstepped %{(fill_ratio*100)-100} removing layer...')
                    botrGen.layers.remove(layer)
                    layers.pop(-1)
                else:
                    break
            elif 1-fill_ratio < 0.1: # move it along
                break
    
# fill a botr until the percentage filled reaches target
def fill_to_target(botrGen: BOTR, dataset: Dataset, search_opts: dict, 
                    target_fill=0.95, render_config: dict=MEDIUM_RENDER, 
                    verbose=True):

    total_px = render_config['outputSize'][0]*render_config['outputSize'][1]
    batch_size = render_config['batchRenderSize']
    max_attempts = 30
    attempts = 0
    last_fill = 0
    while True:
        batch_layers = []
        for _ in range(batch_size):
            example, ann = dataset.get_example_target_area(
                    area_target = search_opts['areaTarget'],
                    area_tolerance = search_opts['areaTolerance'],
                    ann_type = search_opts['ann_type'])
            if get_annotation_supercategory(ann) not in search_opts['exclusions']:
                layer = BOTR_Layer(example, ann)
                batch_layers.append(layer)
                botrGen.append_layer(layer)
                
        rend = botrGen.render(render_config)
        # botrGen.layers.clean_invisible(1)
        # batch render to make everything faster
        for layer in batch_layers:
            if layer.percentage_fill() < 1e-4:
                botrGen.layers.remove(layer)
        total_fill = image_nonzero_px(rend) / total_px

        if total_fill == last_fill:
            attempts += 1
            if attempts > max_attempts:
                return False
        elif attempts > 0:
            attempts -= 1

        last_fill = total_fill
        if verbose:
            print(f'Filling to target: %{(total_fill * 100):.4f}/{target_fill * 100} Layers: {len(botrGen.layers)}')
        if total_fill > target_fill:
            return True

# uses 2 different methods to fill the botr
def render_matching_botr(botrGen: BOTR, dataset: Dataset, child_attrs: dict, inheritence_fill: float=0.75,
                            target_fill: float=0.99, overstep_lim: float=1.25, verbose=False) -> None:
    # fill up comp with matching attributes (with a separate target fill)
    fill_target_attributes_fast(botrGen, dataset,
                child_attrs, comp_target_fill=inheritence_fill, 
                overstep_lim=overstep_lim, verbose=verbose)

    exclusions = [k for k, v in child_attrs.items() if v < 1e-5]
    # clean up any remaining background with random fills
    search_opts = {
        "areaTarget" : 0.05,
        "areaTolerance" : 0.3,
        "ann_type" : "any",
        "exclusions" : exclusions }
    fill_to_target(botrGen, dataset, search_opts, 
                    target_fill, render_config=MEDIUM_RENDER, verbose=verbose)
