# operations on BOTR NFTs including combinations and restructuring
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests

from utils import load_json, save_json, open_image, display_image_meta, sort_dict
from api import metaplex_nft_metadata
# import solana
import numpy as np
from metaplex import reverse_metaplex_attributes, metaplex_attributes, remaining_royalty_share, avg_attributes, sum_attributes, attrs_difference, sort_order_attributes
from rendering import render_matching_botr
from visualization import attribute_breakdown, graph_parent_child_genetics
from botr import BOTR
from config import RENDER_CONFIG_DEFAULT, NFT_CONFIG_DEFAULT
# HOMUNCULI_API
from dataset import Dataset
import collections
# from solana import MintInfo, Token, Client, PublicKey, Keypair

# get_mint_info/

IMG_META_URI = {
    "image" : "out/0.png",
    "metadata" : "out/0.json"
}

class BOTR_NFT():

    def __init__(self, uri_pair: dict=None, address: str=None,
                    parents: list=None):
        if uri_pair is None:
            if address is None:
                return
            self.metadata = metaplex_nft_metadata(address)
            self.image = open_image(self.metadata['image'])
        else:
            self.metadata = load_json(uri_pair['metadata'])
            self.image = open_image(uri_pair['image'])
        self.parents = parents
        self.address = address
        
    def display(self):
        display_image_meta(self.image, self.metadata)

    def category_breakdown(self):
        attribute_breakdown(self.metadata['attributes'], 
                            self.metadata['name'])

    def breed(self, otherNft,
                botrGen: BOTR, nftConfig: dict,
                render_config: dict=RENDER_CONFIG_DEFAULT):
        return generate_child_nft(self, otherNft, botrGen,
                    inheritence_fill=nftConfig['inheritence_fill'],
                    target_fill=nftConfig['target_fill'],
                    overstep_lim=nftConfig['overstep_lim'],
                    render_config=render_config)

    def get_parent_attributes(self, parent: int=0):
        return sort_order_attributes(
            self.parents[parent].metadata['attributes'])

    def get_attributes(self):
        return reverse_metaplex_attributes(self.metadata['attributes'])

    def compare_with_parents(self):
        graph_parent_child_genetics(
            self.get_parent_attributes(0),
            self.get_parent_attributes(1),
            self.get_attributes())

    def get_address(self):
        return self.address
        
    # add a function to upload to metapex candymachine address


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
def generate_child_attrs(nft_1: BOTR_NFT, nft_2: BOTR_NFT):
    attrs_1 = reverse_metaplex_attributes(nft_1.metadata['attributes'])
    attrs_2 = reverse_metaplex_attributes(nft_2.metadata['attributes'])
    child_attrs = make_child_attrs(attrs_1, attrs_2)
    attrs_1 = sort_dict(attrs_1)
    attrs_2 = sort_dict(attrs_2)              
    return child_attrs, attrs_1, attrs_2


def add_parents_to_creators(path: str, parent_1: BOTR_NFT, parent_2: BOTR_NFT,
                                child_attrs: dict = None) -> None:
    metadata = load_json(path)
    creators = NFT_CONFIG_DEFAULT['nft_creators']

    parent_royalties = remaining_royalty_share(creators)
    # TODO:
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
    save_json(path, metadata)

# combine two existing BOTR nfts to generate a new one
def generate_child_nft(nft_1: BOTR_NFT, nft_2: BOTR_NFT,
                            botrGen: BOTR, inheritence_fill: float=0.75,
                            target_fill: float=0.95, overstep_lim: float=1.25,
                            render_config: dict=RENDER_CONFIG_DEFAULT) -> BOTR_NFT:
    child_attrs, attrs_1, attrs_2 = generate_child_attrs(nft_1, nft_2)
    render_matching_botr(botrGen, child_attrs, inheritence_fill, 
                        target_fill, overstep_lim)
    botrGen.layers.clean_invisible(0.001)
    botrGen.generate(render_config)
    png_path, json_path = botrGen.save_assets("out/")
    add_parents_to_creators(json_path, nft_1, nft_2, child_attrs)
    return BOTR_NFT(
        uri_pair={"image":png_path, "metadata":json_path},
        parents=[nft_1, nft_2])

def graph_parent_child_genetics(attrs_parent_1, attrs_parent_2, attrs_child):
    attrs_1_sort = collections.OrderedDict(sorted(attrs_parent_1.items()))
    attrs_2_sort = collections.OrderedDict(sorted(attrs_parent_2.items()))
    child_attr_sort = collections.OrderedDict(sorted(attrs_child.items()))
    keys = list(attrs_1_sort.keys())
    values = range(len(keys))
    plt.figure(figsize=(12,3))
    # plt.subplot(3,1,1)
    plt.plot(attrs_1_sort.values(), label="nft-1", color='red')
    plt.plot(attrs_2_sort.values(), label="nft-2", color='orange')
    plt.plot(child_attr_sort.values(), label="child-generated", color='blue')
    plt.xticks(values, keys, rotation=75)
    plt.legend(loc="upper right")
    plt.show()
    parent_avgs = attrs_difference(attrs_1_sort, attrs_2_sort)
    parent_avgs = {k: (v[0]+v[1])/2 for k, v in parent_avgs.items()}
    plt.figure(figsize=(12,3))
    plt.plot(parent_avgs.values(), label="parent-avg", color='red')
    plt.plot(child_attr_sort.values(), label="child-generated", color='green')
    plt.xticks(values, keys, rotation=75)
    plt.legend(loc="upper right")
    plt.show()

# def metaplex_nft_metadata(address):
#     route = HOMUNCULI_API['routes']['metadata-off-chain']
#     return requests.get(f"{route[0]}{address}/{route[1]}").json()


# get image and metadata uris from solana address
def solana_nft_image_meta(address):
    
    return [None, None]

        # if nft_address is not None:
        #     self.token = Token(
        #         client, 
        #         pubkey, 
        #         program_id, 
        #         payer))

# def combine_two(nft1: BOTR_NFT, nft2: BOTR_NFT):

if __name__ == "__main__":

    botrNft1 = BOTR_NFT({
        "image" : "out/0.png",
        "metadata" : "out/0.json"
    })
    botrNft2 = BOTR_NFT({
        "image" : "out/1.png",
        "metadata" : "out/1.json"
    })

    cocoDataset = Dataset()
    botrGen = BOTR(RENDER_CONFIG_DEFAULT, cocoDataset)
    newNft = generate_child_nft(
        botrNft1, botrNft2, botrGen, 
        inheritance_target_fill = 0.75,
        target_fill=0.95, overstep_lim=1.25)

    # plt.title("Gene Comparison")
    # plt.plot(attrs_1.values(), label="nft 1")
    # plt.plot(attrs_2.values(), label="nft 2")
    # plt.plot(child_attrs.values(), label="child")
    # plt.legend(loc="upper right")
    # plt.show()
    # botrGen.generate(config)
    # botrGen.display()