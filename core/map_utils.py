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
    """
    Wrapper for Maxwell mapping metadata, with electrode-channel conversions and accessors.
    """
    def __init__(self, filepath=None, df=None, from_csv=False, channels=None):
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
            self.mapping = df.copy()

        self.set_mapping(self.mapping)

    @classmethod
    def from_csv(cls, filepath):
        return cls(filepath, from_csv=True)

    @classmethod
    def from_df(cls, df):
        return cls(df=df)

    @classmethod
    def from_maxwell(cls, filepath, channels=None):
        return cls(filepath, channels=channels)

    def set_mapping(self, mapping):
        if mapping is None:
            print("No mapping provided")
            self.mapping = self.channels = self.electrodes = None
            return

        self.mapping = mapping
        self.channels = mapping['channel'].astype(int).tolist()
        self.electrodes = mapping['electrode'].astype(int).tolist()

    def save(self, filepath):
        with smart_open.open(filepath, 'w') as f:
            self.mapping.to_csv(f, index=False)

    # ---------------------------
    # Conversion + Access
    # ---------------------------
    def get_channels(self, electrodes=None):
        if electrodes is None:
            electrodes = self.electrodes
        elif isinstance(electrodes, int):
            electrodes = [electrodes]
        return [
            int(self.mapping[self.mapping['electrode'] == elec]['channel'].values[0])
            for elec in electrodes
        ]

    def get_electrodes(self, channels=None):
        if channels is None:
            channels = self.channels
        elif isinstance(channels, int):
            channels = [channels]
        return [
            int(self.mapping[self.mapping['channel'] == ch]['electrode'].values[0])
            for ch in channels
        ]

    def get_orig_channels(self, channels=None, electrodes=None):
        if channels is not None:
            return [int(self.mapping[self.mapping['channel'] == ch]['orig_channel'].values[0]) for ch in channels]
        elif electrodes is not None:
            return [int(self.mapping[self.mapping['electrode'] == elec]['orig_channel'].values[0]) for elec in electrodes]
        else:
            return self.mapping['orig_channel'].astype(int).tolist()

    def get_positions(self, channels=None, electrodes=None):
        if channels is not None and electrodes is not None:
            raise ValueError("Provide either channels or electrodes, not both.")
        if electrodes is not None:
            channels = self.get_channels(electrodes)
        elif channels is None:
            return self.mapping[['x', 'y']].values

        return self.mapping[self.mapping['channel'].isin(channels)][['x', 'y']].values

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

    def select_electrodes(self, electrodes):
        self.selected_electrodes = electrodes
        self.selected_channels = self.get_channels(electrodes)

    def select_channels(self, channels):
        self.selected_channels = channels
        self.selected_electrodes = self.get_electrodes(channels)


# ==============================
# Visualization Functions
# ==============================

def plot_architecture_map(metadata):
    """
    Show encoder/decoder/training electrodes + neuron positions.
    """
    mapping = metadata['mapping']
    encode_electrodes = metadata['encode_electrodes']
    decode_electrodes = metadata['decode_electrodes']
    training_electrodes = metadata['training_electrodes']
    spike_locs = np.array(metadata['spike_locs'])

    mapper = Mapping.from_df(mapping)
    encode_pos = mapper.get_positions(electrodes=encode_electrodes)
    decode_pos = mapper.get_positions(electrodes=decode_electrodes)
    train_pos = mapper.get_positions(electrodes=training_electrodes)
    all_pos = mapper.get_positions()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(all_pos[:, 0], all_pos[:, 1], s=2, label='Unused', zorder=-3)
    ax.scatter(encode_pos[:, 0], encode_pos[:, 1], c='blue', label='Encode', s=60, marker='X')
    ax.scatter(decode_pos[:, 0], decode_pos[:, 1], c='g', label='Decode', s=60, marker='o',
               facecolors='none', linewidths=1.5)
    ax.scatter(train_pos[:, 0], train_pos[:, 1], c='purple', label='Training', s=60, marker='s',
               facecolors='none', linewidths=1.5)
    ax.scatter(spike_locs[:, 0], spike_locs[:, 1], label='Neural Unit', c='r', alpha=0.6, zorder=-2)

    ax.set_title('Electrode Roles on Array')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_combined_electrode_neuron_map(metadata, show=True, ax=None, figsize=(10, 5)):
    """
    Show neuron and electrode role overlay on same array.
    """
    mapping = metadata['mapping']
    encode_electrodes = metadata.get('encode_electrodes', [])
    decode_electrodes = metadata.get('decode_electrodes', [])
    training_electrodes = metadata.get('training_electrodes', [])
    spike_locs = np.array(metadata.get('spike_locs', []))

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
    ax.scatter(encode_pos[:, 0], encode_pos[:, 1], c='blue', label='Encode', s=60, marker='X')
    ax.scatter(decode_pos[:, 0], decode_pos[:, 1], c='green', label='Decode', s=60,
               marker='o', facecolors='none', linewidths=1.5)
    ax.scatter(train_pos[:, 0], train_pos[:, 1], c='purple', label='Training', s=60,
               marker='s', facecolors='none', linewidths=1.5)
    ax.scatter(spike_locs[:, 0], spike_locs[:, 1], c='red', alpha=0.6, label='Neural Unit', zorder=-2)

    ax.set_title('Electrode Roles on Array')
    ax.set_xlabel('X Position (µm)')
    ax.set_ylabel('Y Position (µm)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax


def plot_electrode_layout(mapping_df):
    """
    Basic scatterplot of all electrode positions.
    """
    x = mapping_df['x'].values
    y = mapping_df['y'].values

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, s=2)
    plt.xlabel('X Position (µm)')
    plt.ylabel('Y Position (µm)')
    plt.title("Electrode Layout")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_neuron_layout(mapping_df, spike_locs):
    """
    Plot electrode layout + neuron spike locations.
    """
    x = mapping_df['x'].values
    y = mapping_df['y'].values
    spike_locs = np.array(spike_locs)

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, s=2, label="Electrodes")
    plt.scatter(spike_locs[:, 0], spike_locs[:, 1], c='r', alpha=0.7, label="Neurons")
    plt.xlabel('X Position (µm)')
    plt.ylabel('Y Position (µm)')
    plt.title("Electrode & Neuron Layout")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


