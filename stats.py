# botr image statistics

from random import shuffle
from nft import Breedable_NFT, generate_child_nft, random_botr_nft
# random_botr_nft
from dataset import Dataset
from botr import BOTR
from config import RENDER_CONFIG_DEFAULT, NFT_CONFIG_DEFAULT

import numpy as np
import matplotlib.pyplot as plt
from rendering import QUICK_RENDER
from scipy.stats import kurtosis
from collections import OrderedDict
from tqdm import tqdm

import threading


# https://docs.scipy.org/doc/scipy/reference/spatial.html


STATS_RENDER = QUICK_RENDER.copy()
STATS_RENDER['outputSize'] = [128,128]


def breed_child(parent_1: Breedable_NFT, parent_2: Breedable_NFT, 
                    dataset: Dataset, rend_config: dict, 
                    history: list, childNum: int) -> Breedable_NFT:
    gen = BOTR(RENDER_CONFIG_DEFAULT)
    child = generate_child_nft(parent_1, parent_2, 
                        gen, dataset, inheritence_fill=0.75,
                        target_fill=0.85, overstep_lim=1.50,
                        render_config=rend_config, verbose=False)
    attrs = gen.get_composition_attributes(rend_config)
    history[childNum] = {"child" : child, "attrs" : attrs }

def breed_loop(dataset: Dataset, loops: int=50, rend_config: dict=STATS_RENDER, base_path: str="assets/", history = {}):
    parent_1 = random_botr_nft(base_path)
    parent_2 = random_botr_nft(base_path)
    for generation in range(loops):
        print(f'generation {generation}/{loops}')
        threads = []
        for i in range(2):
            x = threading.Thread(target=breed_child, args=(
                parent_1, parent_2, dataset, rend_config, history, i+generation))
            threads.append(x)
            x.start()
        for index, thread in enumerate(threads):
            thread.join()
        # they grow up so fast
        parent_1 = history[generation]["child"] 
        parent_2 = history[generation+1]["child"]  
    return history

def shuffle_layers(botrGen: BOTR, runs: dict, shot_num: int, rend_config: dict):
    botrGen.layers.shuffle_order()
    botrGen.generate(rend_config)
    runs[shot_num] = { "item" : botrGen.generatedItem, 
                "attributes" : botrGen.get_composition_attributes(rend_config),
                "dist" : botrGen.layers.fill_distribution(), 
                "layers" : botrGen.layers.get_order() }


def run_shuffles(botrGen: BOTR, batches: int=30, batch_size: int=8,
                    rend_config: dict=STATS_RENDER, runs: dict={}):

    runs[0] = {"item" : botrGen.generatedItem, 
            "attributes" : botrGen.get_composition_attributes(rend_config),
            "dist" : botrGen.layers.fill_distribution(), 
            "layers" : botrGen.layers.get_order() }
    # pbar = tqdm(total=shots)

    for i in range(batches):
        threads = []
        for j in range(batch_size):
            t = threading.Thread(target=shuffle_layers, args=(
                botrGen, runs, (i*batch_size)+j, rend_config ))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    return runs

def std_runs(gens: dict):
    return OrderedDict(
        sorted({ abs(np.std(gen['dist'])): gen for gen in gens }.items()))

def kurtosis_runs(gens: dict):
    return OrderedDict({ (kurtosis(gen['dist'])): gen for gen in gens }.items())

if __name__ == "__main__":
    botrGen = random_botr_nft().get_generator("assets/objects/")
    runs = run_shuffles(botrGen, 30)
    
    std_scores = std_runs(runs)
    kurtosis_scores = kurtosis_runs(runs)

    hi_std = std_scores[min(list(std_scores.keys()))]
    lo_std = std_scores[max(list(std_scores.keys()))]

    hi_kurtosis = kurtosis_scores[min(list(kurtosis_scores.keys()))]
    lo_kurtosis = kurtosis_scores[max(list(kurtosis_scores.keys()))]

    botrGen.layers.set_order(hi_std['layers'])
    botrGen.generate(RENDER_CONFIG_DEFAULT)
    print(f"lowest std {np.std(hi_std['dist'])}")
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(hi_std['dist'])
    plt.subplot(1,2,2)
    plt.imshow(botrGen.generatedItem.image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    botrGen.layers.set_order(lo_std['layers'])
    botrGen.generate(RENDER_CONFIG_DEFAULT)
    print(f"highest std {np.std(lo_std['dist'])}")
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(lo_std['dist'])
    plt.subplot(1,2,2)
    plt.imshow(botrGen.generatedItem.image)
    plt.xticks([])
    plt.yticks([])
    plt.show()