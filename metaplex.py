import json


# convert attributes into list of dicts with trait_type: value format
def format_attributes(dict_attrs):
    attributes = []
    for attr in dict_attrs:
        
        for k, v in attr.items():
            attributes.append( {"trait_type": k, "value": v} )
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


def generate_metadata(
    name,
    royalties, # generated with format_royalty_list
    collection_name='',
    symbol='',
    description='',
    image_file='',
    animation_path='',
    external_url='homunculi.org/art',
    attributes=[],
    files=[],
    category='image',
    collection_family='cymatic-cyborgues',
    seller_fee_basis_points=0):

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

if __name__ == "__main__":
    metadata = generate_metadata(
        name,
        royalties, # generated with format_royalty_list
        collection_name='',
        symbol='',
        description='',
        image_url='',
        animation_url='',
        external_url='homunculi.org/art',
        attributes=[],
        files=[],
        category='image',
        collection_family='cymatic-cyborgues',
        seller_fee_basis_points=0)

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