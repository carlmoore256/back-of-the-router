import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import os

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_coco_categories(path="annotations/stuff_val2017.json"):
    with open(path,'r') as COCO:
        categories = json.loads(COCO.read())['categories']
    categories_id = {}
    for cat in categories:
        categories_id[cat['id']] = cat
    return categories_id

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def copy_file(source, new_directory):
    destination = os.path.join(new_directory, os.path.split(source)[-1])
    shutil.copyfile(source, destination)
    print(f'copied {source} to {destination}')
