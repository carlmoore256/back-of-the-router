# operations on BOTR NFTs including combinations and restructuring
from utils import load_json, open_image, display_image_meta
import solana
# from solana import MintInfo, Token, Client, PublicKey, Keypair

# get_mint_info/

IMG_META_URI = {
    "image" : "out/28.png",
    "metadata" : "out/28.json"
}

class BOTR_NFT():

    def __init__(self, image_meta_uri: dict=None, nft_address: str=None):
        if image_meta_uri is None:
            if nft_address is None:
                return
            image_meta_uri = solana_nft_image_meta()
        self.metadata = load_json(image_meta_uri['metadata'])
        self.image = open_image(image_meta_uri['image'])
        
    def display(self):
        display_image_meta(self.image, self.metadata)
        


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
    nft = BOTR_NFT(IMG_META_URI)
    nft.display()
    # print(nft)