# operations on BOTR NFTs including combinations and restructuring
import matplotlib.pyplot as plt
from utils import load_json, save_json, open_image, display_image_meta, sort_dict
import solana
import numpy as np
from metaplex import reverse_metaplex_attributes, metaplex_attributes
from rendering import render_matching_botr
from botr import BOTR
from config import RENDER_CONFIG_DEFAULT
from dataset import Dataset
import collections
# from solana import MintInfo, Token, Client, PublicKey, Keypair

# get_mint_info/

IMG_META_URI = {
    "image" : "out/0.png",
    "metadata" : "out/0.json"
}

class BOTR_NFT():

    def __init__(self, uri_pair: dict=None, nft_address: str=None,
                    parents: list=None):
        if uri_pair is None:
            if nft_address is None:
                return
            uri_pair = solana_nft_image_meta()
        self.metadata = load_json(uri_pair['metadata'])
        self.image = open_image(uri_pair['image'])
        self.parents = parents
        self.nft_address = nft_address
        
    def display(self):
        display_image_meta(self.image, self.metadata)

    def breed(self, otherNft,
                botrGen: BOTR, nftConfig: dict,
                render_config: dict=RENDER_CONFIG_DEFAULT):
        return generate_child_nft(self, otherNft, botrGen,
                    inheritence_fill=nftConfig['inheritence_fill'],
                    target_fill=nftConfig['target_fill'],
                    overstep_lim=nftConfig['overstep_lim'],
                    render_config=render_config)

    def compare_with_parents(self):
        parent_1_meta = self.parents[0].metadata
        parent_2_meta = self.parents[1].metadata
        graph_parent_child_genetics(
            reverse_metaplex_attributes(parent_1_meta['attributes']),
            reverse_metaplex_attributes(parent_2_meta['attributes']),
            reverse_metaplex_attributes(self.metadata['attributes']))

    def get_address(self):
        return self.nft_address

        
    # add a function to upload to metapex candymachine address

        
def sum_attributes(attrs):
    return sum(list(attrs.values()))

def attrs_difference(attrs_1: dict, attrs_2: dict):
    return {k: [v1, v2] for [k, v1], [_, v2] in zip(attrs_1.items(),attrs_2.items())}

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
    print(f'ATTRS 1 {attrs_1}')
    child_attrs = make_child_attrs(attrs_1, attrs_2)
    attrs_1 = sort_dict(attrs_1)
    attrs_2 = sort_dict(attrs_2)              
    return child_attrs, attrs_1, attrs_2

def add_parents_to_metadata(path: str, parent_1: BOTR_NFT, parent_2: BOTR_NFT) -> None:
    metadata = load_json(path)
    metadata['attributes'] += metaplex_attributes({
        "parent-1" : parent_1.get_address(),
        "parent-2" : parent_2.get_address()})
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
    add_parents_to_metadata(json_path, nft_1, nft_2)

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
    plt.plot(child_attr_sort.values(), label="child-target", color='blue')
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