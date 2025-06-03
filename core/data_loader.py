# data_loader.py - Responsible for loading and organizing spike datasets

import os
import pickle

def load_pickle(path):
    """Load a pickle file from the given path."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def add_label(obj, label):
    """Attach a label to a dataset (as attribute or dict key)."""
    if hasattr(obj, '__dict__'):
        setattr(obj, 'label', label)
    elif isinstance(obj, dict):
        obj['__label__'] = label
    return obj

def load_datasets(paths):
    """Load generic datasets from a dictionary of {label: path}."""
    datasets = {}
    for key, path in paths.items():
        obj = load_pickle(path)
        datasets[key] = add_label(obj, key)
    return datasets

def load_log_data(log_paths):
    """Load multiple log data files, labeling and printing details."""
    log_data = {}
    for label, path in log_paths.items():
        if os.path.exists(path):
            data = load_pickle(path)
            log_data[label] = add_label(data, label)
            print(f"Loaded log data for {label}: {type(data)}, {len(data)} entries")
            print(f"  Keys in {label} log: {list(data.keys())}")
        else:
            print(f"Missing log file for {label}")
    return log_data

def load_spike_data(spike_paths):
    """Load spike data with labels and type info."""
    spike_data = {}
    for label, path in spike_paths.items():
        if os.path.exists(path):
            data = load_pickle(path)
            spike_data[label] = add_label(data, label)
            print(f"Loaded spike data for {label}: {type(data)}")
        else:
            print(f"Missing spike file for {label}")
    return spike_data

def load_causal_info(path="data/causal_info.pkl"):
    """Load and label causal info."""
    if os.path.exists(path):
        data = load_pickle(path)
        print("Loaded causal_info keys:", list(data.keys()))
        return add_label(data, 'causal_info')
    else:
        print("Missing causal_info.pkl")
        return None

def load_metadata(path="data/metadata.pkl"):
    """Load and label metadata."""
    if os.path.exists(path):
        data = load_pickle(path)
        print("Loaded metadata keys:", list(data.keys()))
        return add_label(data, 'metadata')
    else:
        print("Missing metadata.pkl")
        return None
