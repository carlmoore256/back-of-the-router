import json
from collections import OrderedDict
from lib2to3.pgen2.tokenize import generate_tokens
import os
from utils import load_json, save_json, get_all_files, get_asset_pair_path, save_object, check_make_dir, abs_path
from PIL import Image
from nanoid import generate as generate_id
from config import NFT_CONFIG_DEFAULT
import subprocess
import api
# basis points / 100 = percentage

METAPLEX_CONFIG = {
    'symbol' : 'BOTR',
    'description' : 'Algorithmically generated visual confusion',
    'seller_fee_basis_points' : 700,
    'external_url' : 'homunculi.org/art',
    'collection_name' : 'Back-of-the-Router',
    'collection_family' : 'Homunculi',
    'category' : 'image',
    'royalties' : NFT_CONFIG_DEFAULT["nft_creators"],
    'candymachine_cli' : '~/metaplex/js/packages/cli/src/candy-machine-v2-cli.ts',
    'candymachine_config' : 'config/candymachine_config.json'
}


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
# image (png, gif, svg, jpg), video (mp4, mov), audio (mp3, flac, wav), vr (glb, gltf)
FILE_TYPES = {
   "mp4" : "video/mp4",
   "mov" : "video/mov",
   "png" : "image/png",
   "gif" : "image/gif",
   "svg" : "image/svg",
   "jpg" : "image/jpg",
   "mp3" : "audio/mp3",
   "flac" : "audio/flac",
   "wav" : "audio/wav",
   "glb" : "vr/glb",
   "gltf" : "vr/gltf",
   "unknown" : None
}

def load_file(file: str):
    ext = list(FILE_TYPES.keys())[list(FILE_TYPES.values()).index(file['type'])]
    file['type']


def get_file_type(file: str) -> str:
    ext = os.path.splitext(os.path.split(file)[-1])[-1]
    if ext in FILE_TYPES.keys():
        return FILE_TYPES[ext]
    return "unknown"

def format_file_list(files: list) -> list:
    return [{"uri" : f, "type": get_file_type(f)} for f in files]

def format_royalties(creators: list, shares: list) -> list:
    return [{"address" : c, "share": s} for c, s in zip(creators, shares)]

# other properties should be provided as [["property-name", [{prop-prototype},{},...], [], ...]
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
    royalties = [],
    other_properties: list=None) -> dict:

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
    if other_properties is not None:
        for name, property in other_properties:
            metadata["properties"][name] = property
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

def save_metaplex_assets(base_path: str, image: Image, 
                        metadata: dict, product_info: dict=None) -> list:
    idx, [img_path, json_path] = get_asset_pair_path(base_path)
    files = [img_path]

    # contains some data that all homunculi nfts will have
    other_properties = [["homunculi",  {"product-info" : product_info }],]

    metadata = generate_metadata(
        name = f"{METAPLEX_CONFIG['symbol']} #{idx+1:03d}",
        symbol = METAPLEX_CONFIG['symbol'],
        description = f"{metadata['text']['name']}: {metadata['text']['description']}",
        seller_fee_basis_points = METAPLEX_CONFIG['seller_fee_basis_points'],
        image_file = img_path,
        animation_path = '',
        external_url = METAPLEX_CONFIG['external_url'],
        attributes = metaplex_attributes(metadata['composition']),
        collection_name = METAPLEX_CONFIG['collection_name'],
        collection_family = METAPLEX_CONFIG['collection_family'],
        files = [img_path],
        category = METAPLEX_CONFIG['category'],
        royalties = METAPLEX_CONFIG['royalties'],
        other_properties=other_properties)
    image.save(img_path)
    save_json(json_path, metadata)
    
    api.new_asset(product_info['id'], "png", abs_path(img_path), "image")
    api.new_asset(product_info['id'], "json", abs_path(json_path), "metadata")

    print(f"=> Saved image and metadata pair: {img_path} {json_path}")
    return idx, [img_path, json_path], metadata

# deprecated, use save_asset_pair
def save_asset_metadata_pair(path: str, image: Image, metadata: dict):
    index = 0
    while True:
        png_path = os.path.join(path, f"{str(index)}.png")
        json_path = os.path.join(path, f"{str(index)}.json")
        if not os.path.isfile(png_path) and not os.path.isfile(json_path):

            metadata = generate_metadata(
                name = f"{METAPLEX_CONFIG['symbol']} #{index+1:03d}",
                symbol = METAPLEX_CONFIG['symbol'],
                description = f"{metadata['text']['name']}: {metadata['text']['description']}",
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

# change metadata files in a given path to have a royalty config
def change_royalties_existing_meta(asset_path="assets/", 
                    royalties=METAPLEX_CONFIG['royalties']):
    files = get_all_files(asset_path, "json")
    for f in files:
        print(f'modifying royalties for {f}')
        meta = load_json(f)
        meta['properties']['creators'] = royalties
        save_json(f, meta)

def candymachine_upload(asset_path: str="assets/",
    network: str="devnet",
    keypair: str="~/.config/solana/devnet.json",
    config: str=METAPLEX_CONFIG['candymachine_config'],
    cache_id: str="example"):

    command = ['ts-node', 
                METAPLEX_CONFIG['candymachine_cli'],
                'upload',
                '-e', network,
                '-k', keypair,
                '-cp', config,
                '-c', cache_id,
                asset_path]

    proc = subprocess.Popen(command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()

    if stderr:
        return None



    # verify_token_metadata

    # show - shows info on existing metadata, useful for retriving data on other products

    # update_candy_machine