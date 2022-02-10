import numpy as np
import argparse

from botr import BOTR
from dataset import Dataset
from config import RENDER_CONFIG_DEFAULT, DEFAULT_SEARCH_OPTS
from rendering import fill_to_target

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate BOTR image assets')
    parser.add_argument('-o', '--out', type=str, default="assets/",
                    help='outpath for saved image/metadata pairs')
    parser.add_argument('-n', '--num', type=int, default=888,
                    help='number of images to generate')
    args = parser.parse_args()
    cocoDataset = Dataset(subset="coco-safe-licenses")
    search_opts = DEFAULT_SEARCH_OPTS.copy()
    
  
    file_idx = 0
    while file_idx < args.num:
        print(f'generating BOTR #{file_idx}/{args.num}')
        botrGen = BOTR(RENDER_CONFIG_DEFAULT)
        search_opts["areaTarget"] = np.random.uniform(low=0.005, high=0.2, size=(1))
        search_opts["areaTolerance"] =  np.random.uniform(low=0.001, high=0.1, size=(1))
        print(f'=> Search options: areaTarget={search_opts["areaTarget"]} areaTolerance={search_opts["areaTolerance"]}')
        success = fill_to_target(botrGen, cocoDataset, search_opts, target_fill=0.95)
        if not success:
            print(f'[!] No fill solution for attempt, moving on...')
            continue
        botrGen.layers.clean_invisible(0.001)
        botrGen.generate(RENDER_CONFIG_DEFAULT)
        file_idx, _ = botrGen.save_assets(args.out)