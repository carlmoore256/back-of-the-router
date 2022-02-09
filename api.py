import requests
from config import HOMUNCULI_API

def api_base():
    return f"{HOMUNCULI_API['base_url']}:{HOMUNCULI_API['port']}"

def metaplex_nft_metadata(address):
    route = HOMUNCULI_API['routes']['metadata-off-chain']
    return requests.get(f"{api_base()}/{route[0]}{address}/{route[1]}").json()
