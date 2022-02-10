from cv2 import imread, cvtColor, COLOR_BGR2RGB
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from threading import Thread
from zipfile import ZipFile
from tqdm import tqdm
import numpy as np
import requests
import shutil
import pickle
import json
import glob
import os

def run_threaded_fn(func, args):
    t = Thread(target=func, args=args)
    t.start()
    return t

# converts filename to an int index, filename must be formatteds "0.ext", "1.ext"...
def filenum_idx(filename) -> int:
    return int(os.path.splitext(os.path.split(filename)[-1])[0])

def map_assets(base_path: str="assets/"):
    meta = get_all_files(base_path, "json")
    imgs = get_all_files(base_path, "png")
    asset_map = {filenum_idx(im): {"image": im, "metadata" : None} for im in imgs }
    for metadata in meta:
        num = filenum_idx(metadata)
        if num in asset_map.keys():
            asset_map[num]["metadata"] = metadata
        else:
            asset_map[num] = {
                "image" : None,
                "metadata" : metadata }
    return asset_map

# get the lowest number filename
def get_asset_pair_path(base_path: str="assets/"):
    asset_map = map_assets(base_path)
    filekeys = asset_map.keys()
    if len(filekeys) > 0:
        asset_num = max(filekeys)+1
        missing_filenum = [x for x in range(min(filekeys), max(filekeys)) if x not in filekeys]
        if len(missing_filenum) > 0:
            asset_num = missing_filenum[0]
        else:
            asset_num = max(filekeys)+1
    else:
        asset_num = 0
    new_img = os.path.join(base_path, f"{asset_num}.png")
    new_meta = os.path.join(base_path, f"{asset_num}.json")
    return asset_num, [new_img, new_meta]

def get_all_files(path, ext):
    return glob.glob(f'{path}/*.{ext}')

def print_pretty(data):
    print(json.dumps(data, indent=2))

def filter_list(inputList, key, allowedVals):
    filtered = [x[key] for x in inputList if x[key] in allowedVals]
    # for item in inputList:
    #     if item[key] in allowedVals:
    #         filtered.append(item)
    return filtered

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path):
    f = open(path)
    data = json.load(f)
    return data
    
def copy_file(source, new_directory):
    destination = os.path.join(new_directory, os.path.split(source)[-1])
    shutil.copyfile(source, destination)
    print(f'copied {source} to {destination}')

def save_object(dict_obj, path):
    root_path = os.path.split(path)[0]
    if not os.path.exists(root_path):
        print(f'{root_path} does not exist, creating it')
        os.mkdir(root_path)
    with open(path, 'wb') as handle:
        pickle.dump(dict_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(path='dataset/coco_organized.pickle'):
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as handle:
        dict_obj = pickle.load(handle)
    return dict_obj

def sort_dict(data, reverse=True):
    return dict(sorted(data.items(), key=lambda item: item[1], reverse=reverse))

def filter_dict_by(data, key, allowed_vals):
    return dict(filter(lambda elem: elem[1][key] in allowed_vals, data.items()))

def arr2d_to_3d(arr):
    arr = np.expand_dims(np.asarray(arr), -1)
    arr = np.repeat(arr, 3, axis=-1)
    arr = np.repeat(arr, 3, axis=-1)

def arr2d_to_img(arr) -> Image:
    # if arr.dtype == np.float:
    # print(np.max(arr))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def image_nonzero_px(image: Image) -> int:
    img_arr = np.asarray(ImageOps.grayscale(image))
    return np.count_nonzero(img_arr)

def load_image_cv2(path):
    img = imread(path)
    img = cvtColor(img, COLOR_BGR2RGB)
    return img

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def download_file(url, output_path):
    print(f'Downloading file {url}')
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

# Download an image and save into temp dir
def download_image(url, temp_dir="temp/"):
    print(f'Downloading image from {url}')
    name = 'download-temp.png'
    # name = url.split('/')[-1]
    check_make_dir(temp_dir)
    img_path = os.path.join(temp_dir, name)
    download_file(url, img_path)
    return Image.open(img_path)

# open an image either from web or local dir
def open_image(uri):
    if check_if_local(uri):
        return Image.open(uri)
    return download_image(uri)
    # urllib.urlretrieve(url, output_path)

def unzip_file(file_path, output_path):
    print(f'Unzipping {file_path}')
    with ZipFile(file=file_path) as zip_file:
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            zip_file.extract(member=file, path=output_path)

def remove_file(file_path):
    os.remove(file_path)
    print(f'Removed {file_path}')

def check_make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f"made directory {path}")
    return path

def check_if_local(path):
  if path.startswith("http"):
    return False
  return True