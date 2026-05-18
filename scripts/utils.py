import yaml
import os

def LoadJson(file_name, base_path="../configs"):
    JSONPATH = os.path.join(base_path, file_name)
    return yaml.safe_load(open(JSONPATH))