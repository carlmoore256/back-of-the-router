from utils import load_json

CONFIG_PATH = "config/"
DATASET_CONFIG_PATH = "config/dataset_config_val.json"
DATASET_CONFIG = load_json(DATASET_CONFIG_PATH)

GENERATOR_CONFIG_PATH = "config/generator_config.json"
GENERATOR_CONFIG_DEFAULT = load_json(GENERATOR_CONFIG_PATH)

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
