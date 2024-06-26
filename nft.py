# operations on BOTR NFTs including combinations and restructuring
import numpy as np
import collections
import random

from metaplex import reverse_metaplex_attributes, remaining_royalty_share, avg_attributes, \
    attrs_difference, sort_order_attributes, FILE_TYPES
# import metaplex
from utils import load_json, save_json, open_image, sort_dict, map_assets, print_pretty
from visualization import attribute_breakdown, graph_attributes, display_image_meta
from rendering import render_matching_botr
from api import metaplex_nft_metadata, get_product_assets, get_asset_data
from dataset import Dataset
# from botr import BOTR

from config import RENDER_CONFIG_DEFAULT, NFT_CONFIG_DEFAULT


IMG_META_URI = {
    "image" : "out/0.png",
    "metadata" : "out/0.json"
}

class NFT():

    def __init__(self):
        pass

    def from_solana_address(self, address):
        self.metadata = metaplex_nft_metadata(address)
        self.assets = {}
        if 'image' in self.metadata.keys():
            self.assets['image'] = open_image(self.metadata['image'])
        # add functions for loading video, audio, etc


    # def from_metadata_file(self, file):
    #     self.metadata = load_json(file)

    # def load_assets(self, metadata):
    #     assets = {}

    #     if 'image' in self.metadata.keys():
    #         assets['image'] = open_image(metadata['image'])

    #     for f in metadata['properties']['files']:
    #         metaplex.load_file(f)

    #     # for f in metadata['properties']['files']:


    #     for ft in FILE_TYPES:
            
    #         if ft in metadata.keys():
                

    #     if 'image' in metadata.keys():
    #         assets['image'] = open_image(metadata['image'])
            


class Breedable_NFT():

    def __init__(self, product_id: str=None, uri_pair: dict=None, address: str=None,
                    parents: list=None):
        if product_id is not None:
            all_assets = get_product_assets(product_id)
            assert all_assets is not None and len(all_assets) > 0
            self.metadata = get_asset_data(list(filter(lambda x: x['tag'] == "metadata", all_assets))[0]["id"])
            self.image = open_image(list(filter(lambda x: x['tag'] == "image", all_assets))[0]["uri"])
        elif uri_pair is not None:
            self.metadata = load_json(uri_pair['metadata'])
            self.image = open_image(uri_pair['image'])
        elif address is not None:
            self.metadata = metaplex_nft_metadata(address)
            self.image = open_image(self.metadata['image'])

        self.parents = parents
        self.address = address
        
    def display(self):
        display_image_meta(self.image, self.metadata)

    def category_breakdown(self):
        attribute_breakdown(self.metadata['attributes'], 
                            self.metadata['name'])

    def breed(self, otherNft,
                botrGen, dataset: Dataset, nftConfig: dict):
        return generate_child_botr(self, otherNft, botrGen,
                    dataset,
                    inheritence_fill=nftConfig['inheritence_fill'],
                    target_fill=nftConfig['target_fill'],
                    overstep_lim=nftConfig['overstep_lim'])

    def get_parent_attributes(self, parent: int=0):
        return sort_order_attributes(self.parents[parent].metadata['attributes'])

    def get_attributes(self):
        return sort_order_attributes(self.metadata['attributes'])

    def compare_with_parents(self):
        graph_attributes(
            [self.get_parent_attributes(0),
            self.get_parent_attributes(1),
            self.get_attributes()],
            ["parent_1", "parent_2", "child"])
        graph_attributes(
            [avg_attributes(
                self.get_parent_attributes(0),
                self.get_parent_attributes(1)),
            self.get_attributes()],
            ["parent_avg", "child"]) 

    def get_address(self):
        return self.address

    def get_identifier(self):
        return self.metadata['product_info']['id']
        # return self.metadata['properties']['homunculi']['identifier']

    def get_generator(self, base_path="assets/objects/"):
        return get_botr_generator(self.get_identifier(), base_path)
        
    # add a function to upload to metapex candymachine address

def get_botr_generator(identifier: str, base_path: str="assets/objects/"):
    from botr import BOTR
    return BOTR(load_data=f"{base_path}{identifier}.pkl")

# return a similarity score based on the distance, the lower the more similar
def attribute_similarity(attrs_1: dict, attrs_2: dict) -> float:
    diffs = attrs_difference(attrs_1, attrs_2)
    score = sum([abs(v[1] - v[0]) for v in diffs.values()])
    return score

def make_child_attrs(attrs_1: dict, attrs_2: dict, noise_scalar: float=0.01):
    genes = attrs_difference(attrs_1, attrs_2)
    noise = np.random.normal(0, 0.01, (len(genes.values())))
    noise *= np.linspace(1,0,len(noise))
    # create new genes from both parents with added noise
    child_genes = {k: abs((sum(v)/2) + n) for n, [k, v] in zip(noise, genes.items())}
    child_genes = sort_dict(child_genes)
    # normalize the gene pool
    total_vals = sum(list(child_genes.values()))
    child_genes = {k: v/total_vals for k, v in child_genes.items()}
    return child_genes

# cross two nft traits to obtain child attributes of a new one
def generate_child_attrs(nft_1: Breedable_NFT, nft_2: Breedable_NFT):
    attrs_1 = reverse_metaplex_attributes(nft_1.metadata['attributes'])
    attrs_2 = reverse_metaplex_attributes(nft_2.metadata['attributes'])
    child_attrs = make_child_attrs(attrs_1, attrs_2)
    attrs_1 = sort_dict(attrs_1)
    attrs_2 = sort_dict(attrs_2)              
    return child_attrs, attrs_1, attrs_2

def add_parents_to_creators(path: str=None, metadata: str=None, parent_1: Breedable_NFT=None, 
                        parent_2: Breedable_NFT=None, child_attrs: dict = None) -> dict:

    if path is not None:
        metadata = load_json(path)
    if metadata is None:
        print(f'[!] Metadata is none, failed to add parents')
        return
    creators = NFT_CONFIG_DEFAULT['nft_creators']

    parent_royalties = remaining_royalty_share(creators)
    # add contribution depending on how much nft influenced generative properties
    if child_attrs is not None:
        # sort attributes by keys (same for all)
        child_attrs = collections.OrderedDict(sorted(child_attrs.items()))

        parent_1_attrs = sort_order_attributes(parent_1.metadata['attributes'])
        parent_2_attrs = sort_order_attributes(parent_2.metadata['attributes'])
        p1_similarity = attribute_similarity(parent_1_attrs, child_attrs)
        p2_similarity = attribute_similarity(parent_2_attrs, child_attrs)

        p1_points = p1_similarity/(p1_similarity + p2_similarity)
        p2_points = p2_similarity/(p1_similarity + p2_similarity)
        p1_points = int(p1_points * parent_royalties)
        p2_points = int(p2_points * parent_royalties)
        # add remainder to larger share
        remainder = parent_royalties - (p1_points + p2_points)
        if p1_points > p2_points:
            p1_points += remainder
        if p2_points > p1_points:
            p2_points += remainder
        creators.append( { "address" : parent_1.get_address() , "share" : p1_points} )
        creators.append( { "address" : parent_2.get_address() , "share" : p2_points} )
    else:
        for parent in [parent_1, parent_2]:
            creators.append( {"address" : parent.get_address(), 
                "share" : parent_royalties//2} )
    metadata['properties']['creators'] = creators
    if path is not None:
        save_json(path, metadata)
    return metadata

def generate_child_botr(nft_1: Breedable_NFT, nft_2: Breedable_NFT,
                            botrGen, dataset: Dataset, 
                            inheritence_fill: float=0.75,
                            target_fill: float=0.95, overstep_lim: float=1.25):
    child_attrs, attrs_1, attrs_2 = generate_child_attrs(nft_1, nft_2)
    render_matching_botr(botrGen, dataset, child_attrs, inheritence_fill, 
                        target_fill, overstep_lim, False)
    # botrGen.metadata = add_parents_to_creators(metadata=botrGen.metadata, parent_1=nft_1, 
    #                     parent_2=nft_2, child_attrs=child_attrs)
    return botrGen

# combine two existing BOTR nfts to generate a new one
def generate_child_nft(nft_1: Breedable_NFT, nft_2: Breedable_NFT,
                            botrGen, dataset: Dataset, 
                            inheritence_fill: float=0.75,
                            target_fill: float=0.95, overstep_lim: float=1.25,
                            render_config: dict=RENDER_CONFIG_DEFAULT,
                            out_path: str="out/", verbose: bool=False) -> Breedable_NFT:
    child_attrs, attrs_1, attrs_2 = generate_child_attrs(nft_1, nft_2)
    render_matching_botr(botrGen, dataset, child_attrs, inheritence_fill, 
                        target_fill, overstep_lim, verbose)
    botrGen.layers.clean_invisible(0.001)
    botrGen.generate(render_config)
    # if out_path is not None:
    idx, [png_path, json_path] = botrGen.save_assets(out_path, verbose=verbose)
    add_parents_to_creators(path=json_path, parent_1=nft_1, parent_2=nft_2, child_attrs=child_attrs)
    return Breedable_NFT(
        uri_pair={"image":png_path, "metadata":json_path},
        parents=[nft_1, nft_2])

def random_botr_nft(base_path="assets/"):
    assets = map_assets(base_path)
    pair = random.choice(list(assets.values()))
    return Breedable_NFT(uri_pair=pair)


# if __name__ == "__main__":

#     botrNft1 = BOTR_NFT({
#         "image" : "out/0.png",
#         "metadata" : "out/0.json"
#     })
#     botrNft2 = BOTR_NFT({
#         "image" : "out/1.png",
#         "metadata" : "out/1.json"
#     })

#     cocoDataset = Dataset()
#     botrGen = BOTR(RENDER_CONFIG_DEFAULT, cocoDataset)
#     newNft = generate_child_nft(
#         botrNft1, botrNft2, botrGen, 
#         inheritance_target_fill = 0.75,
#         target_fill=0.95, overstep_lim=1.25)