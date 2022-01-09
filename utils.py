import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import os
import pickle

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def display_multiple_images(images=[], titles=[]):
    plt.figure(figsize=(20,10))
    for i in range(len(images)):
        plt.subplot(1,len(images), i+1)
        plt.title(titles[i])
        plt.imshow(images[i])
    plt.show()

def filter_list(inputList, key, allowedVals):
    filtered = [x[key] for x in inputList if x[key] in allowedVals]
    # for item in inputList:
    #     if item[key] in allowedVals:
    #         filtered.append(item)
    return filtered

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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