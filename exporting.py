import os
import numpy as np
import argparse
from utils import check_make_dir, abs_path, save_json, print_pretty, local_uri_to_web, join_paths
# from botr import BOTR_Layer, BOTR
from dataset import get_annotation_supercategory
from config import PROJECT_ID
import api
import cv2
import glob

def save_transparent_layers(botrGen, 
                base_path: str="assets/",
                remove_existing=False):

    layer_objs = []
    id = botrGen.product_info['id']

    outpath =  os.path.join(base_path, "layers")
    check_make_dir(outpath)
    image_path = os.path.join(outpath, "images")
    check_make_dir(image_path)

    for i, layer in enumerate(botrGen.layers):
        # im = layer.raster
        # im = layer.get_image(botrGen.config["outputSize"])
        im = layer.apply_mask(layer.get_raster(full=True))
        rgba = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2RGBA)
        mask = np.asarray(layer.get_mask(botrGen.config["outputSize"])) * 255
        rgba[:,:,3:4] =  mask
        layer_id = f'{i:03d}'
        filename = os.path.join(image_path, f"{id}_{layer_id}.png")
        _, sums, _ = layer.dominant_color()
        cv2.imwrite(filename, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        layer_objs.append( { "uri" : local_uri_to_web(filename), 
                        "fill" : layer.percentage_fill(), 
                        "category" : get_annotation_supercategory(layer.annotation),
                        "colors" : list(sums) } )
    outfile = os.path.join(outpath, f"{id}_layers.json")
    print(f'SAVING {outfile} prev saved img {filename}')
    save_json(outfile, layer_objs)
    if remove_existing:
        api.remove_asset(productId=botrGen.product_info['id'], tag="layers")
    api.new_asset(productId=botrGen.product_info['id'],
            assetType="json", uri=outfile, tag="layers")
    return layer_objs

def export_botr_layers(botr_binary: str, outpath="/var/www/html/botr/nft/assets"):
    from botr import BOTR
    botrGen = BOTR(load_data=botr_binary)
    botrGen.generate()
    botrGen.save_assets(outpath)
    save_transparent_layers(botrGen, outpath)

def clear_export_path(base_dir: str):
    png_files = glob.glob(os.path.join(base_dir, "**/*.png"), recursive=True)
    json_files = glob.glob(os.path.join(base_dir, "**/*.json"), recursive=True)
    pkl_files = glob.glob(os.path.join(base_dir, "**/*.pkl"), recursive=True)
    for f in png_files:
        os.remove(f)
    for f in json_files:
        os.remove(f)
    for f in pkl_files:
        os.remove(f)
    print(f'=> Removed {len(png_files)+len(json_files)+len(pkl_files)} files from {base_dir}')

def batch_export_to_website(binary_dir: str,
                outdir: str, clear_outdir: bool=True):
    if clear_outdir: clear_export_path(outdir)
    api.remove_product(projectId=PROJECT_ID) # clear the database
    botr_files = glob.glob(os.path.join(binary_dir, "*.pkl"))
    for i, f in enumerate(botr_files):
        try:
            print(f'\n=> Exporting file {i+1}/{len(botr_files)}\n')
            export_botr_layers(f, outdir)
        except Exception as e:
            print(f'[!] ERROR:\n{e}\n')

# this is the final export process for botr
# all assets are generated, saved to the website directory, and registered
# in the homunculi database - run script with sudo:
# sudo /home/carl/pyenv/bin/python exporting.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binaries", type=str, default="assets-0.1.0/objects", 
                            required=False, help="location of botr binaries")
    parser.add_argument("--outdir", type=str, default="/var/www/html/botr/nft/assets", 
                            required=False, help="location to save generated assets")
    parser.add_argument("--clear", type=bool, default=True, required=False, 
                            help="delete files from outdir before exporting")
    args = parser.parse_args()
    
    batch_export_to_website(args.binaries, args.outdir, args.clear)