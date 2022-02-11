import numpy as np
import argparse

from botr import BOTR
from dataset import Dataset
from config import RENDER_CONFIG_DEFAULT, DEFAULT_SEARCH_OPTS
from rendering import fill_to_target, SearchOptions
from utils import print_pretty, random_subset
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate BOTR image assets')
    parser.add_argument('-o', '--out', type=str, default="assets/",
                    help='outpath for saved image/metadata pairs')
    parser.add_argument('-n', '--num', type=int, default=888,
                    help='number of images to generate')
    parser.add_argument('-v', '--verbose', type=bool, default=True,
                    help='report events to console')
    args = parser.parse_args()
    cocoDataset = Dataset(subset="coco-safe-licenses")


    # make use of dynamic search options
    search_opts = SearchOptions(dynamic_params = {
        "areaTarget" :  { "func": np.random.uniform, "params" : {"low" : 0.005, "high" : 0.2} },
        "areaTolerance" : { "func": np.random.uniform, "params" : {"low" : 0.001, "high" : 0.1} },
        "posTarget" : { "func": None, "params" : None }, # normalized target
        "posTolerance" : { "func": None, "params" : None },
        "ann_type" :  { "value": "any" },
        "exclusions" :  { "func": random_subset,
                         "params": {"str_list" : ["person", "animal", "vehicle", "sports", "other"], 
                         "exponential" : True} }
    })
    
    file_idx = 0
    while file_idx < args.num:
        botrGen = BOTR(RENDER_CONFIG_DEFAULT)
        opts = search_opts.dynamic_options()

        if args.verbose:
            print(f'=> generating BOTR #{file_idx}/{args.num}\n')

        success = fill_to_target(botrGen, cocoDataset, 
                                opts, 
                                target_fill=0.95)
        if not success:
            if args.verbose:
                print(f'[!] No fill solution for attempt, moving on...')
            continue
        botrGen.layers.clean_invisible(0.001)
        botrGen.generate(RENDER_CONFIG_DEFAULT)
        file_idx, _ = botrGen.save_assets(args.out)