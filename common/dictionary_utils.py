from config import settings as st
import pickle

import json


def new_pickle(outpath, data):
    """(Over)write data to new pickle file."""
    #outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(data, f)
    print(f'Writing new pickle file... {outpath}')


def load_pickle(inpath):
    print(f'Loading from existing pickle file... {inpath}')
    with open(inpath, "rb") as f:
        return pickle.load(f)


def new_json(outpath, data):
    """(Over)write data to new json file."""
    #outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(data, f, indent=4)
    print(f'Writing new json file... {outpath}')


def load_json(inpath):
    print(f'Loading from existing json file... {inpath}')
    with open(inpath, "r") as f:
        return json.load(f)
