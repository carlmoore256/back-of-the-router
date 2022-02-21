# provides product information to any project
# v0.1.0 => date: 10.2.22
import nanoid
from numpy import product
from config import PROJECT_VERSION, PROJECT_ID, API_VERSION
from nanoid import generate as generate_id
# from api import new_product
import api

# deprecated, directly call homunculi api instead
def generate_identifier():
    return generate_id(size=10)

# def api_publish(product_info: dict) -> None:
#     # request api to publish info into database here
#     return None

# def publish_product(productId: str, projectId: str=PROJECT_ID, 
#                         projectVersion: str=PROJECT_VERSION):
#     product_info = {
#         "product-id" : productId,
#         "project-id" : projectId,
#         "project-version" : projectVersion,
#         "api-version" : API_VERSION,
#         "homunculi-id" : productId }
#     api_publish(product_info)
#     return product_info


def new_product(productId: str=None, projectId: str=PROJECT_ID, 
                        projectVersion: str=PROJECT_VERSION) -> dict:
    print(f'PRODUCT ID {productId}')
    if productId is None:
        # default to having the api request new info
        info = api.new_product(projectId=projectId)
        if info is not None:
            return api.get_product_info(info["id"])
            
        # in the case of offline
        info = {
            "id": generate_identifier(),
            "projectId": projectId,
            "mintAddress": None,
            "metadataUri": None }
        return info

    # try to get info on an existing product
    info = api.get_product_info(productId)
    if info is not None:
        return info

    # if product is not yet registered with the database, register it
    info = api.new_product(projectId, productId)
    # get the product from the database
    print(f'=> Created new product {info}')
    return api.get_product_info(info["id"])