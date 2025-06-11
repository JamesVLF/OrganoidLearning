# data_loader.py - Responsible for loading and organizing spike datasets

import os
import pickle
from pathlib import Path
from spikedata.spikedata import SpikeData
import numpy as np
import zipfile
import scipy.io as sio

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

def load_curation(qm_path):
    """
    Load spike data from a curation zip file (acqm).

    Parameters:
    qm_path (str): Path to the .zip file containing spike data.

    Returns:
    tuple: (train, neuron_data, config, fs)
        train (list): List of spike times arrays (seconds).
        neuron_data (dict): Neuron metadata.
        config (dict or None): Configuration dictionary if present.
        fs (float): Sampling frequency.
    """
    with zipfile.ZipFile(qm_path, 'r') as f_zip:
        qm = f_zip.open("qm.npz")
        data = np.load(qm, allow_pickle=True)
        spike_times = data["train"].item()
        fs = data["fs"]
        train = [times / fs for _, times in spike_times.items()]
        config = data["config"].item() if "config" in data else None
        neuron_data = data["neuron_data"].item()
    return train, neuron_data, config, fs

'''
acqm_path = "./23126c_D44_KOLFMO_5272025_acqm.zip"

# Load data from acqm file
train_acqm, neuron_data, config, fs = load_curation(acqm_path)
sd_acqm = SpikeData(train_acqm)

embed()
'''

def extract_zip(zip_filename, base_folder, target_folder, label):
    """
    Extracts a ZIP file and renames the extracted `.npz` file to match the experiment label.
    """
    # coerce to Path
    base_folder   = Path(base_folder)
    target_folder = Path(target_folder)
    zip_filename  = Path(zip_filename)

    zip_path = base_folder / zip_filename
    if not zip_path.exists():
        print(f"File not found: {zip_path}")
        return None

    target_folder.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        z.extractall(target_folder)

    npz_files = [m for m in members if m.endswith(".npz")]
    if not npz_files:
        print("! No .npz in", zip_path.name)
        return None

    src = target_folder / npz_files[0]
    dst = target_folder / f"{label}.npz"
    src.replace(dst)
    print("Renamed", npz_files[0], "→", dst.name)
    return dst

def normalize_conditions(conditions, data_dict):
    if conditions is None:
        return list(data_dict.keys())
    elif isinstance(conditions, str):
        return [conditions] if conditions in data_dict else []
    elif isinstance(conditions, list):
        return [c for c in conditions if c in data_dict]
    else:
        return []

def load_npz_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    spike_times_sec = {neuron_id: spikes / data["fs"] for neuron_id, spikes in data["train"].item().items()}
    return spike_times_sec, data["neuron_data"].item(), data["fs"]

def load_all_data(file_dict, conditions=None):
    selected_conditions = normalize_conditions(conditions, file_dict)

    if not selected_conditions:
        print("No valid dataset(s) found. Available options:", list(file_dict.keys()))
        return {}

    loaded_data = {}
    for condition in selected_conditions:
        print(f"Loading {condition} ...")
        spike_times_sec, neuron_data, fs = load_npz_data(file_dict[condition])
        loaded_data[condition] = {
            "spike_times_sec": spike_times_sec,
            "neuron_data": neuron_data,
            "fs": fs
        }
        print(f"Loaded {len(spike_times_sec)} neurons for {condition}.")

    print("\nSelected datasets successfully loaded.\n")
    return loaded_data

def mat_to_spikeData(mat_path):
    mat = sio.loadmat(mat_path)
    units = [i[0][0]*1e3 for i in mat['spike_times']]
    sd = SpikeData(units)
    return sd

def inspect_dataset(condition, data_dict):
    """
    Prints basic information about a spike dataset.

    Parameters:
    - condition (str): Dataset key to inspect (e.g., 'Baseline').
    - data_dict (dict): Dictionary containing SpikeData instances.
    """
    if condition not in data_dict:
        print(f"[ERROR] Condition '{condition}' not found in provided data.")
        return

    sd = data_dict[condition]
    try:
        unit_ids, spike_times = sd.idces_times()
    except Exception as e:
        print(f"[WARNING] Could not extract unit IDs or spike times: {e}")
        unit_ids, spike_times = [], []

    print("\n--- Dataset Inspection ---")
    print(f"Condition           : {condition}")
    print(f"Number of Neurons   : {sd.N}")
    print(f"Recording Length    : {sd.length / 1000:.2f} seconds")
    print(f"Sample Unit IDs     : {unit_ids[:10]}")
    print(f"Sample Spike Times  : {spike_times[:10]}")
    print(f"Available Attributes: {dir(sd)}")
    print("-----------------------------")


def inspect_datasets(conditions=None, data_dict=None):
    """
    Inspects one, multiple, or all datasets.

    Parameters:
    - conditions (str, list, or None): Dataset condition(s) to inspect.
    - data_dict (dict): Dictionary containing all loaded datasets.
    """
    if not data_dict:
        print("No datasets loaded.")
        return

    # Normalize input
    conditions = normalize_conditions(conditions, data_dict)

    for condition in conditions:
        inspect_dataset(condition, data_dict)
        print("-" * 50)
