# data_loader.py - Responsible for loading and organizing spike datasets
import pickle

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_datasets(paths):
    datasets = {}
    for key, path in paths.items():
        datasets[key] = load_pickle(path)
    return datasets
