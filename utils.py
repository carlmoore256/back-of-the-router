import numpy as np
import matplotlib.pyplot as plt
import json

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_coco_categories(path="annotations/instances_val2017.json"):
    with open(path,'r') as COCO:
        categories = json.loads(COCO.read())['categories']
    categories_id = {}
    for cat in categories:
        categories_id[cat['id']] = cat
    return categories_id

def save_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)