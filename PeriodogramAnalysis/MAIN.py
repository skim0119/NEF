from miv.core.operator import DataLoader
from miv.io.openephys import DataManager
from miv.core.pipeline import Pipeline
from PeriodogramAnalysis import PeriodogramAnalysis
from PowerSpectrumAnalysis import SpectrumAnalysis

# save_path = "results/"
# path:str = load_data(progbar_disable=True).data_collection_path

path: str = "/Users/aia/Downloads/2024-08-25_19-49-12"
# path: str = "/home1/10197/qxwang/BAKS_test/2024-08-25_19-49-12"
print('file path:', path)

dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

Periodogram_Analysis = PeriodogramAnalysis(
        exclude_channels=[ch for ch in range(128) if ch not in [2, 3]],
        win_sec=4,
        band=[[0.5, 4], [12, 30]],
        mark_region = True
    )

Spectrum_Analysis = SpectrumAnalysis(
        exclude_channels=[ch for ch in range(128) if ch not in [3]],
        frequency_limit=[0, 10],
        win_sec=4,
        nperseg = 100,
        noverlap= 50,
        band_display = [0, 2],
        convert_db = False,
    )

data >> Periodogram_Analysis
data >> Spectrum_Analysis
pipeline1 = Pipeline(Periodogram_Analysis)
pipeline2 = Pipeline(Spectrum_Analysis)
# pipeline1.run(working_directory="results/", verbose=True)
pipeline2.run(working_directory="results/", verbose=True)

