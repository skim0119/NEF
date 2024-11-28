from miv.io.openephys import DataManager
from miv.core.pipeline import Pipeline
from spectrum_analysis import PowerSpectrumAnalysis
from spectrogram_analysis import SpectrogramAnalysis
from miv.core.operator import DataLoader
from miv.core.operator import Operator
from miv.datasets.openephys_sample import load_data
from power_density_statistics import (
    SpectrumAnalysisPeriodogram,
    SpectrumAnalysisWelch,
)


# Download the sample data
path: str = load_data(progbar_disable=True).data_collection_path
print(path)

working_directory = "results"

dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

spectrum_welch = SpectrumAnalysisWelch()
spectrum_per = SpectrumAnalysisPeriodogram()
Periodogram_Analysis = PowerSpectrumAnalysis()
Spec_Analysis = SpectrogramAnalysis()

data >> spectrum_welch >> Periodogram_Analysis
data >> spectrum_per
data >> Spec_Analysis

pipeline1 = Pipeline(Periodogram_Analysis)
pipeline2 = Pipeline(spectrum_per)
pipeline3 = Pipeline(Spec_Analysis)
pipeline1.run(working_directory=working_directory, verbose=True)
pipeline2.run(working_directory=working_directory, verbose=True)
pipeline3.run(working_directory=working_directory, verbose=True)
