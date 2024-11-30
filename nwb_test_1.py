import os

import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

from miv.core.datatype import Signal, Spikestamps
from power_density_statistics import (
    SpectrumAnalysisPeriodogram,
    SpectrumAnalysisWelch,
)

file_path = "/Users/aia/Downloads/RecordNode103__experiment1__recording1.nwb"

# Generate lfp signal
def lfp_signal_generator(lfp_series):
    sampling_rate = lfp_series.rate
    num_chunks = lfp_series.data.shape[0]

    for chunk in range(num_chunks):
        lfp_data = lfp_series.data[chunk, :, :]

        timestamps = np.linspace(
            chunk * len(lfp_data) / sampling_rate,
            (chunk + 1) * len(lfp_data) / sampling_rate,
            len(lfp_data)
        )

        yield Signal(
            data=lfp_data,
            timestamps=timestamps,
            rate=sampling_rate
        )

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

    # lfp_interface = nwbfile.processing["ecephys"].data_interfaces["LFP"]
    # lfp_series = lfp_interface.electrical_series["ElectricalSeries"]
    #
    # lfp_gen = lfp_signal_generator(lfp_series)
    #
    # chunk_limit = 0
    # channel_limit = 0
    # chunk = 0
    #
    # spectrum_welch = SpectrumAnalysisWelch()
    # result = spectrum_welch(lfp_gen)
    #
    # signal_summary_file_path = "./signal_analysis"
    # for signal_chunk in lfp_gen:
    #     for channel in range(channel_limit):
    #         plt.figure(figsize=(10, 4))
    #         plt.plot(signal_chunk.timestamps, signal_chunk.data[:, channel])
    #         plt.title("LFP Signal")
    #         plt.xlabel("Time (s)")
    #         plt.ylabel("Amplitude")
    #         plt.grid()
    #         plot_path = os.path.join(signal_summary_file_path, f"lfp_figure_chunk{signal_chunk}_channel{channel}.png")
    #         os.makedirs(signal_summary_file_path, exist_ok=True)
    #         plt.savefig(plot_path, dpi=300)
    #         plt.close()
    #
    #     chunk += 1
    #     if chunk == channel_limit:
    #         break

    # Open spike data
    spike_series = nwbfile.acquisition["Spike Events"]
    print(spike_series)

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
        plt.legend(loc='upper right')
        plot_path = os.path.join(spike_summary_file_path, f"spike_train_chunk{chunk}_all_channels.png")
        os.makedirs(spike_summary_file_path, exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()


        chunk += 1
        if chunk == channel_limit:
            break
