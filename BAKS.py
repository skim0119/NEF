import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import os
from scipy.special import gamma
from miv.core.operator import Operator, DataLoader
from miv.core.pipeline import Pipeline
from miv.io.openephys import Data, DataManager
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.signal.spike import ThresholdCutoff

# Download the sample data
path: str = "/Users/aia/Downloads/2024-08-25_19-49-12"
print(path)

# Create data modules:
dataset: DataManager = DataManager(data_collection_path=path)
print(f"Dataset length: {len(dataset)}")  # Check the length of the dataset

data: DataLoader = dataset[0]
# Create operator modules:
bandpass_filter: Operator = ButterBandpass(lowcut=300, highcut=3000, order=4)
lfp_filter: Operator = ButterBandpass(highcut=3000, order=2, btype='lowpass')
spike_detection: Operator = ThresholdCutoff(cutoff=4.0, progress_bar=True)

# Build analysis pipeline
data >> bandpass_filter >> spike_detection
# data >> lfp_filter

# print(data.summarize())  # Print the summary of data flow

pipeline = Pipeline(spike_detection)  # Create a pipeline object to get `spike_detection` output
pipeline.run(working_directory="results/", verbose=True)  # Save outcome into "results" directory
print(pipeline.summarize())

spike_signal = next(iter(spike_detection.output()))  # Next is used to retrieve the first fragment of the output
spike_signal = np.array(spike_signal)
y_values = np.zeros(len(spike_signal))

# save the spiketime data
directory = 'spike'
file_path = os.path.join(directory, 'spike_times')
os.makedirs(directory, exist_ok=True)
np.savetxt(file_path, spike_signal, delimiter=',', header='spike times', comments='')

# BAKS
def BAKS(SpikeTimes, Time, a, b):
    N = len(SpikeTimes)
    sumnum = 0
    sumdenum = 0
    # Calculate the adaptive bandwidth h
    for i in range(N):
        numerator = (((Time - SpikeTimes[i]) ** 2) / 2 + 1 / b) ** -a
        denumerator = (((Time - SpikeTimes[i]) ** 2) / 2 + 1 / b) ** -(a + 0.5)
        sumnum += numerator
        sumdenum += denumerator

    h = (gamma(a) / gamma(a + 0.5)) * (sumnum / sumdenum)

    # Estimate the firing rate
    FiringRate = 0
    for j in range(N):
        K = (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-((Time - SpikeTimes[j]) ** 2) / (2 * h ** 2))
        FiringRate += K

    return FiringRate, h

# calculate firing rate with the best a
Time = 1660 - spike_signal[0]
a = 0.64
b = len(spike_signal) ** (4./5.)

FiringRate, h = BAKS(spike_signal - spike_signal[0], Time, a, b)
print("Firing Rate:", FiringRate)

index = []
rate = []
i = 0
for a in np.arange (0.1, 1.8, 0.01):  # try different a and find the a with minimum MISE
    b = len(spike_signal) ** (4. / 5.)
    FiringRate, h = BAKS(spike_signal - spike_signal[0], Time, a, b)
    index.append(a)
    rate.append(FiringRate.mean())
    i += i

index = np.array(index)
rate = np.array(rate)

plt.figure(figsize=(10, 2))
plt.plot(index, rate)
plt.xlabel('a value')
plt.ylabel('firing rate')
plt.show()