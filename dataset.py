# contains functions for downloading the coco dataset
import argparse
import os
from utils import download_file, unzip_file, load_json, remove_file, check_make_path, print_pretty
from env import DATASET_CONFIG

def download_dataset_archives(config):
    for filename, url in config["downloads"].items():
        print(f"\n====== Retrieving {filename} ====")
        zip_path = os.path.join(config["temp_path"], filename)
        download_file(url, zip_path)
        unzip_file(zip_path, config["base_path"])
        remove_file(zip_path)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Initialize BOTR COCO Dataset')
    # parser.add_argument('--config', type=str, required=False, default="config/dataset_config.json", 
    #                     help='path to the dataset config json')
    # args = parser.parse_args()
    config = load_json(DATASET_CONFIG)
    check_make_path(config["base_path"])
    check_make_path(config["temp_path"])
    download_dataset_archives(config)