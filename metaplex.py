import json
from collections import OrderedDict
import os
from utils import save_json
from config import METAPLEX_CONFIG
from PIL import Image

# changes a dictionary with values into the metaplex format:
# [ {"trait_type" : v, "value" : v} , {}, ... ]
def metaplex_attributes(data):
    attributes = []
    for k, v in data.items():
        attributes.append({"trait_type" : k, "value" : v})
    return attributes

# changes dict of type [ {"trait_type" : v, "value" : v} , {}, ... ]
# back into {attribute: value}
def reverse_metaplex_attributes(data):
    attributes = {}
    for attr in data:
        attributes[attr['trait_type']] = attr['value']
    return attributes

def remaining_royalty_share(royalties):
    total = 0
    for creator in royalties:
        total += creator["share"]
    return 100-total

# def calculate_royalty_points():


def format_file_list(files):
    f_list = []
    for f in files:
        f_type = None
        if f.endswith("mp4"):
            f_type = "video/mp4"
        if f.endswith("png"):
            f_type = "image/png"
        f_list.append({"uri" : f, "type": f_type})
    return f_list

def format_royalties(creators, shares):
    c_list = []
    for c, s in zip(creators, shares):
        c_list.append({
            "address": c,
            "share": s
        })
    return c_list

def format_botr_metaplex(botr_data, image_uri, animation_url, website_url="homunculi.org/art"):
    attributes = metaplex_attributes(botr_data['composition'])
    return attributes


def generate_metadata(
    name = '',
    symbol = '',
    description = '',
    seller_fee_basis_points = 7,
    image_file = '',
    animation_path = '',
    external_url = 'homunculi.org/art',
    attributes = [],
    collection_name = '',
    collection_family = 'Homunculi',
    files = [],
    category = 'image',
    royalties = []):

    file_list = format_file_list(files)

    metadata = {
        "name": name,
        "symbol":symbol,
        "description": description,
        "seller_fee_basis_points": seller_fee_basis_points,
        "image": image_file,
        "animation_url": animation_path,
        "external_url": external_url,
        "attributes": attributes,

        "collection": {
            "name": collection_name,
            "family": collection_family
        },
        "properties": {
            "files": file_list,
            "category": category,
            "creators": royalties
        }
    }
    return metadata

# =============== Helpers ====================================

def avg_attributes(attrs_1: dict, attrs_2: dict) -> dict:
    diff = attrs_difference(attrs_1, attrs_2)
    return {k: (v[0]+v[1])/2 for k, v in diff.items()}

def sum_attributes(attrs: dict) -> float:
    return sum(list(attrs.values()))

def attrs_difference(attrs_1: dict, attrs_2: dict) -> dict:
    return {k: [v1, v2] for [k, v1], [_, v2] in zip(attrs_1.items(),attrs_2.items())}

# sort attributes by name value
def sort_order_attributes(metaplex_attrs) -> OrderedDict:
    attrs = reverse_metaplex_attributes(metaplex_attrs)
    return OrderedDict(sorted(attrs.items()))

def save_asset_metadata_pair(path: str, image: Image, metadata: dict):
    index = 0
    while True:
        png_path = os.path.join(path, f"{str(index)}.png")
        json_path = os.path.join(path, f"{str(index)}.json")
        if not os.path.isfile(png_path) and not os.path.isfile(json_path):

            metadata = generate_metadata(
                name = metadata['text']['name'],
                symbol = METAPLEX_CONFIG['symbol'],
                description = METAPLEX_CONFIG['description'],
                seller_fee_basis_points = METAPLEX_CONFIG['seller_fee_basis_points'],
                image_file = png_path,
                animation_path = '',
                external_url = METAPLEX_CONFIG['external_url'],
                attributes = metaplex_attributes(metadata['composition']),
                collection_name = METAPLEX_CONFIG['collection_name'],
                collection_family = METAPLEX_CONFIG['collection_family'],
                files = [png_path],
                category = METAPLEX_CONFIG['category'],
                royalties = METAPLEX_CONFIG['royalties'])
            image.save(png_path)
            save_json(json_path, metadata)
            print(f"saved image and metadata pair: {png_path} {json_path}")
            break
        else:
            index += 1
    return png_path, json_path
# =============== Interfaces ==================================


# METAPLEX_ATTRS = {
#     'symbol' : 'BOTR',
#     'description' : 'Confusing images',
#     'seller_fee_basis_points' : 0,
#     'external_url' : 'homunculi.org/art',
#     'collection_name' : 'Back-of-the-Router',
#     'collection_family' : 'Homunculi',
#     'category' : 'image',
#     'royalties' : format_royalties(["6TNtaPn8MEaBvekb7PzFnPTm6aTHdvFmSBoRznjETjXK"],[100]) 
# }


if __name__ == "__main__":
    royalties = format_royalties(
        ["6TNtaPn8MEaBvekb7PzFnPTm6aTHdvFmSBoRznjETjXK"], [100])
    metadata = generate_metadata(
        name = '',
        symbol = '',
        description = '',
        seller_fee_basis_points = 0,
        image_file = '',
        animation_path = '',
        external_url = 'homunculi.org/art',
        attributes = [],
        collection_name = '',
        collection_family = 'Homunculi',
        files = [],
        category = 'image',
        royalties = royalties)