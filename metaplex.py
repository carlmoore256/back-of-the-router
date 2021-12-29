import json


# convert attributes into list of dicts with trait_type: value format
def format_attributes(dict_attrs):
    attributes = []
    for attr in dict_attrs:
        for k, v in attr.items():
            attributes.append( {"trait_type": k, "value": v} )
    return attributes

def generate_metadata(
    name, 
    symbol, 
    description, 
    external_url='homunculi.org/art',
    seller_fee_basis_points=0):


    metadata = {
        "name": "Solflare X NFT",
        "symbol": "",
        "description": "Celebratory Solflare NFT for the Solflare X launch",
        "seller_fee_basis_points": 0,
        "image": "https://www.arweave.net/abcd5678?ext=png",
        "animation_url": "https://www.arweave.net/efgh1234?ext=mp4",
        "external_url": "https://solflare.com",
        "attributes": [
            {
            "trait_type": "web",
            "value": "yes"
            },
            {
            "trait_type": "mobile",
            "value": "yes"
            },
            {
            "trait_type": "extension",
            "value": "yes"
            }
        ],
        "collection": {
            "name": "Solflare X NFT",
            "family": "Solflare"
        },
        "properties": {
            "files": [
            {
                "uri": "https://www.arweave.net/abcd5678?ext=png",
                "type": "image/png"
            },
            {
                "uri": "https://watch.videodelivery.net/9876jkl",
                "type": "unknown",
                "cdn": true
            },
            {
                "uri": "https://www.arweave.net/efgh1234?ext=mp4",
                "type": "video/mp4"
            }
            ],
            "category": "video",
            "creators": [
            {
                "address": "xEtQ9Fpv62qdc1GYfpNReMasVTe9YW5bHJwfVKqo72u",
                "share": 100
            }
            ]
        }
        }

    for k, v in dict_attrs.items():
        metadata