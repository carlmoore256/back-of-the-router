from utils import load_json

CONFIG_PATH = "config/"
DATASET_CONFIG_PATH = "config/dataset_config.json"
DATASET_CONFIG = load_json(DATASET_CONFIG_PATH)

GENERATOR_CONFIG_PATH = "config/generator_config.json"
GENERATOR_CONFIG_DEFAULT = load_json(GENERATOR_CONFIG_PATH)


