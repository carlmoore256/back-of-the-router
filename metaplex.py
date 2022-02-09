import json

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
    formatted = {}
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

METAPLEX_ATTRS = {
    'symbol' : 'BOTR',
    'description' : 'Confusing images',
    'seller_fee_basis_points' : 0,
    'external_url' : 'homunculi.org/art',
    'collection_name' : 'Back-of-the-Router',
    'collection_family' : 'Homunculi',
    'category' : 'image',
    'royalties' : format_royalties(["6TNtaPn8MEaBvekb7PzFnPTm6aTHdvFmSBoRznjETjXK"],[100]) 
}

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

# "attributes": [
#     {
#     "trait_type": "web",
#     "value": "yes"
#     },
#     {
#     "trait_type": "mobile",
#     "value": "yes"
#     },
#     {
#     "trait_type": "extension",
#     "value": "yes"
#     }
# ],

# "files": [
# {
#     "uri": "https://www.arweave.net/abcd5678?ext=png",
#     "type": "image/png"
# },
# {
#     "uri": "https://www.arweave.net/efgh1234?ext=mp4",
#     "type": "video/mp4"
# }
# ]