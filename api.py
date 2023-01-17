from importlib_metadata import metadata
from numpy import product
import requests
from nanoid import generate as generate_id
import json
from utils import local_uri_to_web, abs_path

HOMUNCULI_API = {
  "base_url" : "https://homunculi.org",
  "port" : 7743,
#   "projectId" : "KrN3BUb8eM",
  "routes" : {
    "metadata" : "api/metaplex/",
    "metadata-off-chain" : ["api/metaplex/", "off-chain"],
    "new-project" : "api/nft/new-project",
    "new-product" : "api/nft/new-product",
    "new-asset" : "api/nft/new-asset",
    "project" : "api/nft/project",
    "product" : "api/nft/product",
    "asset" : "api/nft/asset",
    "remove-product" : "api/nft/remove-product",
    "remove-asset" : "api/nft/remove-asset",
    "product-assets" : ["api/nft/product/", "/assets"],
    "asset-data" : "api/nft/asset/",
    "update-breed-request" : "api/botr/update-breed-request",
    "breed-requests" : "api/botr/breed-requests",
    "collection-size" : ["api/nft/project/", "/largest-sequence-number"]
  }
}

# deprecated, directly call homunculi api instead
# def generate_identifier():
#     return generate_id(size=10)

# if the length is 0, return none
def handle_post_result(res):
    if len(res) == 0:
        return None
    if len(res) == 1:
        return res[0]
    return res

def api_base():
    return f"{HOMUNCULI_API['base_url']}:{HOMUNCULI_API['port']}/"

# def get(request):
#     print(f'REQUEST {request}')
#     try:
#         return requests.get(request).json()
#     except Exception as e:
#         print(f'[!] request {request} failed!\n{e}')
#         return None
def get(route):
    request = f"{api_base()}{route}"
    # print(f'REQUEST {request}')
    try:
        return requests.get(request).json()
    except Exception as e:
        print(f'[!] request {request} failed!\n{e}')
        return None

def post(route, data):
    request = f"{api_base()}{route}"
    try:
        return handle_post_result(
            requests.post(request, json=data).json())
    except Exception as e:
        print(f'[!] request {request} failed!\n{e}')
        return None 

def metaplex_nft_metadata(address):
    route = HOMUNCULI_API['routes']['metadata-off-chain']
    return get(f"{route[0]}{address}/{route[1]}")

# ======================================
# ************** DATABASE **************
# ======================================

# == New Entries ==

def new_project(name: str, description: str, 
                current_version="0.1.0") -> dict:
    post_data = { "name" : name,
                "description" : description, 
                "currentVersion" : current_version }
    route = HOMUNCULI_API['routes']['new-project']
    return post(route, post_data)

def new_product(projectId: str, productId: str=None) -> dict:
    route = HOMUNCULI_API['routes']['new-product']
    post_data = { "projectId" : projectId }
    if productId is not None:
        post_data["id"] = productId
    return post(route, post_data)

def new_asset(productId: str, assetType: str, uri: str, tag: str=None) -> dict:
    route = HOMUNCULI_API['routes']['new-asset']
    # checks if uri is in the web directory
    # converts path to the full absolute path
    uri = local_uri_to_web(uri)
    post_data = {
        "productId": productId,
        "type": assetType,
        "uri": uri,
        "tag" : tag }
    return post(route, post_data)

# == Get Helpers ==

def get_project_info(projectId: str) -> dict:
    post_data = { "id" : projectId }
    route = HOMUNCULI_API['routes']['project']
    return post(route, post_data)

def get_product_info(productId: str) -> dict:
    route = HOMUNCULI_API['routes']['product']
    post_data = { "id" : productId }
    return post(route, post_data)

# == Full Queries ==

def query_products(productId: str=None, projectId: str=None,
            mintAddress: str=None, metadataUri: str=None):
    route = HOMUNCULI_API['routes']['product']
    post_data = format_query(
        [productId, projectId, mintAddress, metadataUri],
        ["id", "projectId", "mintAddress", "metadataUri"])
    return post(route, post_data)

def query_assets(assetId: str=None, productId: str=None, 
            assetType: str=None, uri: str=None, tag: str=None):
    route = HOMUNCULI_API['routes']['asset']
    post_data = format_query(
        [assetId, productId, assetType, uri, tag],
        ["id", "productId", "type", "uri", "tag"])
    return post(route, post_data)

# == Remove Queries ==

def remove_product(productId: str=None, projectId: str=None, 
            mintAddress: str=None, metadataUri: str=None) -> dict:
    route = HOMUNCULI_API['routes']['remove-product']
    post_data = format_query(
        [productId, projectId, mintAddress, metadataUri],
        ["id", "projectId", "mintAddress", "metadataUri"])
    print(f'REMOVING post {post_data}')
    return post(route, post_data)

def remove_asset(assetId: str=None, productId: str=None, 
            assetType: str=None, uri: str=None, tag: str=None) -> dict:
    route = HOMUNCULI_API['routes']['remove-asset']
    post_data = format_query(
        [assetId, productId, assetType, uri, tag],
        ["id", "productId", "type", "uri", "tag"])
    return post(route, post_data)

# cleanly removes product and associated assets
def remove_product_and_assets(productId: str):
    remove_asset(productId=productId)
    remove_product(productId=productId)

# remove all assets for project cleanly
def remove_project_products_and_assets(projectId: str):
    # get all project asset
    remove_product(projectId=projectId)
    # products = query_products(projectId=projectId)
    # for p in products:
    #     remove_product(productId=p["id"])

def remove_product_asset_by_tag(productId: str, tag: str):
    remove_asset(productId=productId, tag=tag)

def get_product_assets(productId: str):
    route = HOMUNCULI_API['routes']['product-assets']
    return get(f"{route[0]}{productId}{route[1]}")

def get_asset_data(assetId: str):
    route = HOMUNCULI_API['routes']['asset-data']
    return get(f"{route}{assetId}")
    
# == Query Helpers ==

# format multiple arguments into a query, drop none
def format_query(args: list, names: list):
    assert len(args) == len(names)
    query = {}
    for arg, name in zip(args, names):
        if arg is not None:
            query[name] = arg
    return query

# =============================================

def request_breed_status(status: str):
    route = HOMUNCULI_API['routes']['breed-requests']
    # gonna return an array of requests
    #  return status, id, data
    return get(f"{route}/{status}")

def complete_breed_request(data):
    route = HOMUNCULI_API['routes']['update-breed-request']
    return post(route, data)

def get_collection_size(projectId: str):
    route = HOMUNCULI_API['routes']['collection-size']
    return get(f"{route[0]}{projectId}{route[1]}")["sequenceNumber"]
