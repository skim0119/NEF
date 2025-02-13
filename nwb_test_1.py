import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

from miv.core.datatype import Signal, Spikestamps

file_path = "/Users/skim0119/Results/RecordNode103__experiment1__recording1.nwb"


# Generate spike train
def spike_train_generator(spike_series, segment_length=60):
    spike_timestamps = spike_series.timestamps[:]
    max_time = spike_timestamps.max()
    num_channels = spike_series.data.shape[1]
    start_time = 0

    while start_time < max_time:
        end_time = start_time + segment_length

        # Get spike index for each chunk
        start_idx = np.searchsorted(spike_timestamps, start_time, side="left")
        end_idx = np.searchsorted(spike_timestamps, end_time, side="right")

        # Generate a matrix that contain every timestamp of each channel
        time_matrix = []

        # Gather timestamps for a channel
        for channel in range(num_channels):
            channel_timestamps = spike_timestamps[start_idx:end_idx]
            channel_spikes = spike_series.data[start_idx:end_idx, channel]

            # Only save spike time in channel_spikes
            valid_mask = channel_spikes > 0
            valid_timestamps = channel_timestamps[valid_mask]

            time_matrix.append(valid_timestamps)

        yield Spikestamps(time_matrix)

        start_time += segment_length


with NWBHDF5IO(file_path, "r") as io:
    nwbfile = io.read()
    print("Top-level keys in NWB file:", nwbfile.fields.keys())
    print("Top-level keys in NWB file:", nwbfile.processing.keys())
    print("Top-level keys in NWB file:", nwbfile.acquisition.keys())

    # print(nwbfile.electrode_groups)
    # print(nwbfile.electrodes)
    print(nwbfile.processing)

    # Open spike data
    spike_series = nwbfile.acquisition["Spike Events"]
    print(spike_series)

    breakpoint()

    data = spike_series.data[:].T
    time = spike_series.timestamps[:]

    plt.imshow(data, cmap="gray_r", extent=(time[0], time[-1], 1, data.shape[0]))
    plt.gca().set_aspect("auto")
    plt.xlim(time[0], time[0] + 10)
    plt.show()

    sys.exit()

    spike_gen = spike_train_generator(spike_series)
    chunk_limit = 2
    channel_limit = 2
    chunk = 0

    # Plot spike data
    spike_summary_file_path = "./spike_analysis"
    for spike_chunk in spike_gen:
        for channel, spike_times in enumerate(spike_chunk):
            y_values = [channel] * len(spike_times)
            plt.scatter(spike_times, y_values, s=1, color="blue")

        plt.title(f"Spike Train - All Channels (Chunk {chunk})")
        plt.xlabel("Time (s)")
        plt.ylabel("Channel Index")
        plt.legend(loc="upper right")
        plot_path = os.path.join(
            spike_summary_file_path, f"spike_train_chunk{chunk}_all_channels.png"
        )
        os.makedirs(spike_summary_file_path, exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()

        chunk += 1
        if chunk == channel_limit:
            break
