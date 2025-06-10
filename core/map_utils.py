import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from braindance.analysis import data_loader

try:
    from braingeneers.utils import smart_open_braingeneers as smart_open
    bgr = True
except ImportError:
    import smart_open
    print("Could not import smart_open_braingeneers")
    bgr = False


# ==============================
# Mapping Class
# ==============================
class Mapping:
    def __init__(self, filepath=None, df=None, from_csv=False, channels=None):
        """Initialize Mapping from a file or DataFrame."""
        self.mapping = None
        self.selected_electrodes = []
        self.selected_channels = []

        if filepath:
            if from_csv:
                with smart_open.open(filepath, 'r') as f:
                    self.mapping = pd.read_csv(f)
            else:
                self.mapping = data_loader.load_mapping_maxwell(filepath, channels)
        elif isinstance(df, pd.DataFrame):
            self.mapping = df

        self.set_mapping(self.mapping)

    # ---- Factory methods ----
    @classmethod
    def from_csv(cls, filepath):
        return cls(filepath, from_csv=True)

    @classmethod
    def from_df(cls, df):
        return cls(df=df)

    @classmethod
    def from_maxwell(cls, filepath, channels=None):
        return cls(filepath, channels=channels)

    # ---- Selection methods ----
    def select_electrodes(self, electrodes):
        self.selected_electrodes = electrodes
        self.selected_channels = [
            int(self.mapping[self.mapping['electrode'] == e]['channel'].values[0])
            for e in electrodes
        ]

    def select_channels(self, channels):
        self.selected_channels = channels
        self.selected_electrodes = [
            int(self.mapping[self.mapping['channel'] == ch]['electrode'].values[0])
            for ch in channels
        ]

    # ---- Configuration ----
    def set_mapping(self, mapping):
        if mapping is None:
            print("No mapping provided")
            self.mapping = self.channels = self.electrodes = None
            return

        self.mapping = mapping
        self.channels = mapping['channel'].astype(int).tolist()
        self.electrodes = mapping['electrode'].astype(int).tolist()

    # ---- Conversion ----
    def get_electrodes(self, channels=None):
        channels = self.channels if channels is None else [channels] if isinstance(channels, int) else channels
        return [
            int(self.mapping[self.mapping['channel'] == ch]['electrode'].values[0])
            for ch in channels
        ]

    def get_channels(self, electrodes=None):
        electrodes = self.electrodes if electrodes is None else [electrodes] if isinstance(electrodes, int) else electrodes
        return [
            int(self.mapping[self.mapping['electrode'] == elec]['channel'].values[0])
            for elec in electrodes
        ]

    def get_orig_channels(self, channels=None, electrodes=None):
        if channels is not None:
            return [int(self.mapping[self.mapping['channel'] == ch]['orig_channel'].values[0]) for ch in channels]
        elif electrodes is not None:
            return [int(self.mapping[self.mapping['electrode'] == elec]['orig_channel'].values[0]) for elec in electrodes]
        else:
            return self.mapping['orig_channel'].astype(int).tolist()

    def get_nearest(self, channel=None, electrode=None, n=None, distance=None):
        if channel is None and electrode is None:
            raise ValueError("Must provide either channel or electrode")
        if electrode is not None:
            channel = self.get_channels(electrode)[0]

        pos = self.mapping[self.mapping['channel'] == channel][['x', 'y']].values[0]
        all_positions = self.mapping[['channel', 'x', 'y']].values
        distances = np.linalg.norm(all_positions[:, 1:] - pos, axis=1)

        channel_distances = [
            (int(all_positions[i][0]), distances[i])
            for i in range(len(distances)) if int(all_positions[i][0]) != channel
        ]
        channel_distances.sort(key=lambda x: x[1])

        if distance is not None:
            channel_distances = [cd for cd in channel_distances if cd[1] <= distance]

        nearest_channels = [ch for ch, _ in channel_distances[:n] if n is not None or True]
        return self.get_electrodes(nearest_channels) if electrode is not None else nearest_channels

    def get_positions(self, channels=None, electrodes=None):
        if channels is not None and electrodes is not None:
            raise ValueError("Provide either channels or electrodes, not both.")
        if electrodes is not None:
            channels = self.get_channels(electrodes)
        elif channels is None:
            return self.mapping[['x', 'y']].values

        return self.mapping[self.mapping['channel'].isin(channels)][['x', 'y']].values




    # ---- Save ----
    def save(self, filepath):
        with smart_open.open(filepath, 'w') as f:
            self.mapping.to_csv(f, index=False)




# ==============================
# Visualization Functions
# ==============================
def plot_architecture_map(metadata):
    mapping = metadata['mapping']
    encode_electrodes = metadata['encode_electrodes']
    decode_electrodes = metadata['decode_electrodes']
    training_electrodes = metadata['training_electrodes']
    spike_locs = np.array(metadata['spike_locs'])

    mapper = Mapping.from_df(mapping)
    encode_positions = mapper.get_positions(electrodes=encode_electrodes)
    decode_positions = mapper.get_positions(electrodes=decode_electrodes)
    training_positions = mapper.get_positions(electrodes=training_electrodes)
    all_positions = mapper.get_positions()

    spikes_x = spike_locs[:, 0]
    spikes_y = spike_locs[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(all_positions[:, 0], all_positions[:, 1], s=2, label='Unused', zorder=-3)
    ax.scatter(encode_positions[:, 0], encode_positions[:, 1], c='blue', label='Encode',
               s=60, marker='X', alpha=1)
    ax.scatter(decode_positions[:, 0], decode_positions[:, 1], c='g', label='Decode',
               s=60, marker='o', alpha=0.8, facecolors='none', linewidths=1.5)
    ax.scatter(training_positions[:, 0], training_positions[:, 1], c='purple', label='Training',
               s=60, marker='s', alpha=0.8, facecolors='none', linewidths=1.5)
    ax.scatter(spikes_x, spikes_y, label='Neural Unit', c='r', alpha=0.6, zorder=-2)

    ax.set_title('Electrode Roles on Array')
    ax.set_xlabel('X Position (µm)')
    ax.set_ylabel('Y Position (µm)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def plot_combined_electrode_neuron_map(metadata, show=True, ax=None, figsize=(10, 5)):
    """
    Plots a combined map of electrode roles and spike-localized neurons.
    """
    mapping = metadata['mapping']
    encode_electrodes = metadata.get('encode_electrodes', [])
    decode_electrodes = metadata.get('decode_electrodes', [])
    training_electrodes = metadata.get('training_electrodes', [])
    spike_locs = np.array(metadata.get('spike_locs', []))

    if spike_locs.ndim != 2 or spike_locs.shape[1] != 2:
        raise ValueError("metadata['spike_locs'] must be a list or array of (x, y) coordinates")


    mapper = Mapping.from_df(mapping)
    encode_pos = mapper.get_positions(electrodes=encode_electrodes)
    decode_pos = mapper.get_positions(electrodes=decode_electrodes)
    train_pos = mapper.get_positions(electrodes=training_electrodes)
    all_pos = mapper.get_positions()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(all_pos[:, 0], all_pos[:, 1], s=2, label='Unused', zorder=-3)
    ax.scatter(encode_pos[:, 0], encode_pos[:, 1], c='blue', label='Encode', s=60, marker='X', alpha=1)
    ax.scatter(decode_pos[:, 0], decode_pos[:, 1], c='green', label='Decode', s=60,
               marker='o', facecolors='none', linewidths=1.5)
    ax.scatter(train_pos[:, 0], train_pos[:, 1], c='purple', label='Training', s=60,
               marker='s', facecolors='none', linewidths=1.5)
    ax.scatter(spike_locs[:, 0], spike_locs[:, 1], c='red', alpha=0.6, label='Neural Unit', zorder=-2)

    ax.set_title('Electrode Roles on Array')
    ax.set_xlabel('X Position (µm)')
    ax.set_ylabel('Y Position (µm)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


