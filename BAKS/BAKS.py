import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from scipy.special import gamma
from miv.core.operator import Operator, DataLoader
from miv.core.pipeline import Pipeline
from miv.io.openephys import Data, DataManager
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.signal.spike import ThresholdCutoff
from dataclasses import dataclass
from miv.core.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call

# Download the sample data
path: str = "/Users/aia/Downloads/2024-08-25_19-49-12"
print(path)

# Create data modules:
dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

# Create operator modules:
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4, tag="bandpass")
spike_detection: Operator = ThresholdCutoff(cutoff=4.0, dead_time=0.002, tag="spikes", progress_bar=True)

data >> bandpass_filter >> spike_detection

@dataclass
class BAKS_firing_rate(OperatorMixin):
    tag = "BAKS firing rate"
    def __post_init__(self):
        super().__init__()

    # Call: important algorithmic steps
    @cache_call
    def __call__(self, spike_detection):
        # BAKS
        def BAKS(SpikeTimes, Time, a, b):
            N = len(SpikeTimes)
            sumnum = 0
            sumdenum = 0
            # Calculate the adaptive bandwidth h
            for i in range(N):
                basicterm = (((Time - SpikeTimes[i]) ** 2) / 2 + 1 / b)
                numerator = basicterm ** -a
                denumerator = basicterm ** -(a + 0.5)
                sumnum += numerator
                sumdenum += denumerator

            h = (gamma(a) / gamma(a + 0.5)) * (sumnum / sumdenum)

            # Estimate the firing rate
            FiringRate = 0
            for j in range(N):
                K = (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-((Time - SpikeTimes[j]) ** 2) / (2 * h ** 2))
                FiringRate += K

            return FiringRate, h

        spike_times = spike_detection.output()
        spikestamps_view = spike_times.get_view(0, 2000)

        self.firing_rate_list = []

        for ch in range(0, len(spikestamps_view)):
            spikestamp = spikestamps_view[ch]

            # Calculate firing rate using BAKS function
            Time = spikestamp[-1] - spikestamp[0]
            a = 0.32
            b = len(spikestamp) ** (4. / 5.)
            FiringRate, h = BAKS(spikestamp - spikestamp[0], Time, a, b)
            rate_ref = len(spikestamp) / Time
            self.firing_rate_list.append((ch, FiringRate, rate_ref))

        self.firing_rate_list.sort(key=lambda x: x[1], reverse=True)

    # after_call process
    def after_run_print(self, output):

        # Get channel information and firing rate from BAKS and Metric
        channel = [item[0] for item in self.firing_rate_list]
        rate = [item[1] for item in self.firing_rate_list]

        file_path = "results/spikes/firing_rate_histogram.csv"
        df = pd.read_csv(file_path)

        rate_em_ordered = df.set_index('channel').loc[channel].reset_index()['firing_rate_hz']

        # Plot comparison
        plt.figure(figsize=(15,5))
        plt.errorbar(channel, rate, xerr=0.2, fmt='none', ecolor='blue', capsize=2, label='BAKS_rate')
        plt.errorbar(channel, rate_em_ordered, xerr=0.2, fmt='none', ecolor='red', capsize=2, label='Metric')
        for i in range(len(channel)):
            plt.vlines(channel[i], rate[i], rate_em_ordered[i], colors='gray',
                       label='Connection' if i == 0 else "")
        plt.xlabel('channels')
        plt.ylabel('firing rate')
        plt.legend()
        plt.show()

        # Calculate difference between Metric and BAKS
        output = 0
        for i in range(len(channel)):
            output += np.sqrt((rate[i] - rate_em_ordered[i]) ** 2)
        print(output)

        # Save data
        directory = 'spike'
        os.makedirs(directory, exist_ok=True)

        summary_file_path = os.path.join(directory, 'firing_rate_summary_sorted.txt')

        with open(summary_file_path, 'w') as summary_file:
            summary_file.write("channel, firing_rate_hz, ref_firing_rate_spikes/time\n")

            for ch, firing_rate, rate_ref in self.firing_rate_list:
                summary_file.write(f"{ch}, {firing_rate}, {rate_ref}\n")

        return output

# System structure
if __name__ == '__main__':
    firing_rate = BAKS_firing_rate()
    spike_detection >> firing_rate

    pipeline = Pipeline(firing_rate)
    pipeline.run(working_directory="results/", verbose=True)