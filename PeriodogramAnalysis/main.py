from miv.io.openephys import DataManager
from miv.core.pipeline import Pipeline
from spectrum_analysis import PowerSpectrumAnalysis
from miv.core.operator import DataLoader
from power_density_statistics import (
    SpectrumAnalysisPeriodogram,
    SpectrumAnalysisMultitaper,
    SpectrumAnalysisWelch,
)
from miv.datasets.openephys_sample import load_data

# Download the sample data
path:str = load_data(progbar_disable=True).data_collection_path
print(path)

path: str = "/Users/aia/Downloads/2024-08-25_19-49-12"
print('file path:', path)

working_directory = "results"

dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

spectrum_welch = SpectrumAnalysisWelch()
spectrum_per = SpectrumAnalysisPeriodogram()
spectrum_mul = SpectrumAnalysisMultitaper()
Periodogram_Analysis = PowerSpectrumAnalysis()

data >> spectrum_welch
data >> spectrum_per
data >> spectrum_mul
pipeline1 = Pipeline(spectrum_welch)
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
