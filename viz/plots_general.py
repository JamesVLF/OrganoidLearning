# plots_general.py - Generic visualizations for spike data
import matplotlib.pyplot as plt

def plot_raster(spike_data, title="Spike Raster", start=0, end=None):
    idces, times = spike_data.idces_times()
    if end is None:
        end = spike_data.length / 1000
    mask = (times >= start * 1000) & (times <= end * 1000)
    times, idces = times[mask], idces[mask]

    plt.figure(figsize=(15, 6))
    plt.title(title)
    plt.scatter(times / 1000, idces, marker='|', s=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Unit #")
    plt.show()

def plot_firing_rates(rates, title="Mean Firing Rates"):
    plt.figure(figsize=(8, 4))
    plt.hist(rates, bins=30)
    plt.title(title)
    plt.xlabel("Firing Rate (Hz)")
    plt.ylabel("Number of Neurons")
    plt.show()

def plot_correlation_matrix(matrix, title="Correlation Matrix"):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.title(title)
    plt.colorbar(label="Correlation")
    plt.show()
