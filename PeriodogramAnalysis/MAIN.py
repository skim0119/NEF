from miv.core.operator import DataLoader
from miv.io.openephys import DataManager
from miv.core.pipeline import Pipeline
from PeriodogramAnalysis import PeriodogramAnalysis
from PowerSpectrumAnalysis import SpectrumAnalysis
from miv.datasets.openephys_sample import load_data

# save_path = "results/"
# path:str = load_data(progbar_disable=True).data_collection_path

path: str = "/Users/aia/Downloads/2024-08-25_19-49-12"
# path: str = "/home1/10197/qxwang/BAKS_test/2024-08-25_19-49-12"
print('file path:', path)

dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

Periodogram_Analysis = PeriodogramAnalysis(
        exclude_channel_list=[ch for ch in range(128) if ch not in [3]],
        window_length_for_welch=4
    )

Spectrum_Analysis = SpectrumAnalysis(
        exclude_channel_list=[ch for ch in range(128) if ch not in [3]],
        window_length_for_welch=4,
        frequency_limit=[0.5, 4],
        band_display = [0, 5]
    )

data >> Periodogram_Analysis
data >> Spectrum_Analysis
pipeline1 = Pipeline(Periodogram_Analysis)
pipeline2 = Pipeline(Spectrum_Analysis)
pipeline1.run(working_directory="results/", verbose=True)
# pipeline2.run(working_directory="results/", verbose=True)

