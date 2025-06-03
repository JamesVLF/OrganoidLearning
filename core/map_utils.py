import numpy as np
import pandas as pd

import braindance
from braindance.analysis import data_loader

try:
    from braingeneers.utils import smart_open_braingeneers as smart_open
    bgr = True
except ImportError:
    import smart_open
    print("Could not import smart_open_braingeneers")
    bgr = False


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
    def from_csv(cls, filepath):
        return cls(filepath, from_csv=True)

    def from_df(cls, df):
        return cls(df=df)

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
    # ---- Mapping configuration ----
    def set_mapping(self, mapping):
        """Stores mapping and flattens lists of channels/electrodes."""
        if mapping is None:
            print("No mapping provided")
            self.mapping = self.channels = self.electrodes = None
            return

        self.mapping = mapping
        self.channels = mapping['channel'].astype(int).tolist()
        self.electrodes = mapping['electrode'].astype(int).tolist()


    # ---- Conversion methods ----
    def get_electrodes(self, channels=None):
        """Return electrode(s) corresponding to given channel(s)."""
        channels = self.channels if channels is None else [channels] if isinstance(channels, int) else channels
        return [
            int(self.mapping[self.mapping['channel'] == ch]['electrode'].values[0])
            for ch in channels
        ]
    def get_channels(self, electrodes=None):
        """Return channel(s) corresponding to given electrode(s)."""
        electrodes = self.electrodes if electrodes is None else [electrodes] if isinstance(electrodes, int) else electrodes
        return [
            int(self.mapping[self.mapping['electrode'] == elec]['channel'].values[0])
            for elec in electrodes
        ]
    def get_orig_channels(self, channels=None, electrodes=None):
        """Return original channels for given channels or electrodes."""
        if channels is not None:
            return [int(self.mapping[self.mapping['channel'] == ch]['orig_channel'].values[0]) for ch in channels]
        elif electrodes is not None:
            return [int(self.mapping[self.mapping['electrode'] == elec]['orig_channel'].values[0]) for elec in electrodes]
        else:
            return self.mapping['orig_channel'].astype(int).tolist()

    # ---- Spatial methods ----
    def get_nearest(self, channel=None, electrode=None, n=None, distance=None):
        """
        Get nearest channels (or electrodes) based on physical distance.
        Prioritize 'n' closest or those within given 'distance'.
        """
        if channel is None and electrode is None:
            raise ValueError("Must provide either channel or electrode")

        if electrode is not None:
            channel = self.get_channels(electrode)[0]

        pos = self.mapping[self.mapping['channel'] == channel][['x', 'y']].values[0]
        all_positions = self.mapping[['channel', 'x', 'y']].values
        distances = np.linalg.norm(all_positions[:, 1:] - pos, axis=1)

        channel_distances = [
            (int(all_positions[i][0]), distances[i])
            for i in range(len(distances))
            if int(all_positions[i][0]) != channel
        ]

        channel_distances.sort(key=lambda x: x[1])

        if distance is not None:
            channel_distances = [cd for cd in channel_distances if cd[1] <= distance]

        nearest_channels = [ch for ch, _ in channel_distances[:n] if n is not None or True]

        return self.get_electrodes(nearest_channels) if electrode is not None else nearest_channels

    def get_positions(self, channels=None, electrodes=None):
        """Return x, y positions for specified channels or electrodes."""
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
