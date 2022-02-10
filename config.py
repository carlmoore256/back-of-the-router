from utils import load_json

CONFIG_PATH = "config/"

DATASET_CONFIG_PATH = "config/dataset_config.json"
DATASET_CONFIG = load_json(DATASET_CONFIG_PATH)

GENERATOR_CONFIG_PATH = "config/generator_config.json"
GENERATOR_CONFIG_DEFAULT = load_json(GENERATOR_CONFIG_PATH)

# a production-ready version of render config for generating final assets
RENDER_CONFIG_PATH = "config/render_config.json"
RENDER_CONFIG_DEFAULT = load_json(RENDER_CONFIG_PATH)

# parameters for combining and generating new NFTs
NFT_CONFIG_PATH = "config/nft_config.json"
NFT_CONFIG_DEFAULT = load_json(NFT_CONFIG_PATH)

# lstm language mode
LSTM_CONFIG_PATH = "config/lstm_config.json"
LSTM_CONFIG = load_json(LSTM_CONFIG_PATH)

HOMUNCULI_API = {
  "base_url" : "https://homunculi.org",
  "port" : 7743,
  "routes" : {
    "metadata" : "api/metaplex/",
    "metadata-off-chain" : ["api/metaplex/", "off-chain"]
  }
}

DEFAULT_SEARCH_OPTS = {
    "areaTarget" : 0.07,
    "areaTolerance" : 0.01,
    "posTarget" : [0.5, 0.5], # normalized target
    "posTolerance" : 1,
    "ann_type" : "any",
    "exclusions" : ["person", "animal", "vehicle", "sports", "other"]
}

# basis points / 100 = percentage

METAPLEX_CONFIG = {
    'symbol' : 'BOTR',
    'description' : 'Algorithmically generated visual confusion',
    'seller_fee_basis_points' : 700,
    'external_url' : 'homunculi.org/art',
    'collection_name' : 'Back-of-the-Router',
    'collection_family' : 'Homunculi',
    'category' : 'image',
    'royalties' : NFT_CONFIG_DEFAULT["nft_creators"]
}

# all available coco supercategories
SUPERCATEGORIES = [
  "ceiling",
  "appliance",
  "vehicle",
  "raw-material",
  "sky",
  "sports",
  "furniture",
  "floor",
  "textile",
  "kitchen",
  "water",
  "person",
  "electronic",
  "furniture-stuff",
  "wall",
  "building",
  "outdoor",
  "indoor",
  "ground",
  "structural",
  "window",
  "solid",
  "animal",
  "food",
  "other",
  "plant",
  "accessory",
  "food-stuff"
]
