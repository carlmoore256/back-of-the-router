from curses import meta
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import os
import pickle
import cv2
import glob

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

def save_dict(dict_obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(dict_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(path='dataset/coco_organized.pickle'):
    with open(path, 'rb') as handle:
        dict_obj = pickle.load(handle)
    return dict_obj

def sort_dict(data, reverse=True):
    return dict(sorted(data.items(), key=lambda item: item[1], reverse=reverse))

def arr2d_to_3d(arr):
    arr = np.expand_dims(np.asarray(arr), -1)
    arr = np.repeat(arr, 3, axis=-1)
    arr = np.repeat(arr, 3, axis=-1)

def load_image_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_asset_metadata_pair(path, image, metadata):
    index = 0
    while True:
        png_path = os.path.join(path, f"{str(index)}.png")
        json_path = os.path.join(path, f"{str(index)}.json")
        if not os.path.isfile(png_path) and not os.path.isfile(json_path):
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
