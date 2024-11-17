from miv.core.operator import DataLoader
from miv.io.openephys import DataManager
from miv.core.pipeline import Pipeline
from spectrum_analysis import PowerSpectrumAnalysis
from spectrogram_analysis import SpectrogramAnalysis
from miv.datasets.openephys_sample import load_data
from miv.signal.filter import ButterBandpass, MedianFilter
from miv.core.operator import Operator, DataLoader
from power_density_statistics import (
    SpectrumAnalysisPeriodogram,
    SpectrumAnalysisMultitaper,
    SpectrumAnalysisWelch,
)

working_directory = "results"
path: str = "/Users/aia/Downloads/2024-08-25_19-49-12"
# path: str = "/home1/10197/qxwang/BAKS_test/2024-08-25_19-49-12"
print("file path:", path)

dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

spectrum_welch = SpectrumAnalysisWelch()
spectrum_per = SpectrumAnalysisPeriodogram()
spectrum_mul = SpectrumAnalysisMultitaper()
Periodogram_Analysis = PowerSpectrumAnalysis()

data >> spectrum_welch >> Periodogram_Analysis
data >> spectrum_per
data >> spectrum_mul
pipeline1 = Pipeline(Periodogram_Analysis)
pipeline2 = Pipeline(spectrum_per)
pipeline3 = Pipeline(spectrum_mul)
pipeline1.run(working_directory=working_directory, verbose=True)
pipeline2.run(working_directory=working_directory, verbose=True)
pipeline3.run(working_directory=working_directory, verbose=True)
# data >> Periodogram_Analysis
# data >> Spectrum_Analysis
# pipeline1 = Pipeline(Periodogram_Analysis)
# pipeline2 = Pipeline(Spectrum_Analysis)
# pipeline1.run(working_directory=working_directory, verbose=True)
# pipeline2.run(working_directory=working_directory, verbose=True)
