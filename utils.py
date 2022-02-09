from curses import meta
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import os
import pickle
import cv2
import glob
# import urllib
import requests
from zipfile import ZipFile
from tqdm import tqdm
from PIL import Image, ImageOps
from metaplex import format_file_list, generate_metadata, metaplex_attributes, format_file_list, METAPLEX_ATTRS

def print_pretty(data):
    print(json.dumps(data, indent=2))

def imshow(img, title='', size=(10,10)):
    plt.figure(figsize=size)
    plt.title(title)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def display_multiple_images(images=[], titles=[], size=(20,10)):
    plt.figure(figsize=size)
    for i in range(len(images)):
        plt.subplot(1,len(images), i+1)
        plt.title(titles[i])
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()
    plt.close()

def display_image_meta(image, metadata, size=(10,10)):
    imshow(
        image, 
        f"{metadata['symbol']} : {metadata['name']}",
        size)
    print_pretty(metadata)

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
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def image_nonzero_px(image: Image) -> int:
    img_arr = np.asarray(ImageOps.grayscale(image))
    return np.count_nonzero(img_arr)

def load_image_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_asset_metadata_pair(path, image, metadata, metaplex=True):


    index = 0
    while True:
        png_path = os.path.join(path, f"{str(index)}.png")
        json_path = os.path.join(path, f"{str(index)}.json")
        if not os.path.isfile(png_path) and not os.path.isfile(json_path):
            if metaplex:
                metadata = generate_metadata(
                    name = metadata['text']['name'],
                    symbol = METAPLEX_ATTRS['symbol'],
                    description = METAPLEX_ATTRS['description'],
                    seller_fee_basis_points = METAPLEX_ATTRS['seller_fee_basis_points'],
                    image_file = png_path,
                    animation_path = '',
                    external_url = METAPLEX_ATTRS['external_url'],
                    attributes = metaplex_attributes(metadata['composition']),
                    collection_name = METAPLEX_ATTRS['collection_name'],
                    collection_family = METAPLEX_ATTRS['collection_family'],
                    files = [png_path],
                    category = METAPLEX_ATTRS['category'],
                    royalties = METAPLEX_ATTRS['royalties'])
            image.save(png_path)
            save_json(json_path, metadata)
            print(f"saved image and metadata pair: {png_path} {json_path}")
            break
        else:
            index += 1

    # all_pngs = glob.glob(f"{path}/*.png")
    # all_jsons = glob.glob(f"{path}/*.json")

# def generate_numbered_filename(path, ext):
#     all_files = glob.glob(f"{path}/*{ext}")

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
        unit_divisor=1024,
    ) as bar:
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

def check_if_local(path):
  if path.startswith("http"):
    return False
  return True
    # with zipfile.ZipFile(file_path, 'r') as zip_ref:
    #     zip_ref.extractall(output_path)

# https://stackoverflow.com/questions/13530762/how-to-know-bytes-size-of-python-object-like-arrays-and-dictionaries-the-simp
def get_obj_size(obj):
    import gc
    import sys

    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz