import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sps
import scipy.signal as sps_sig
from typing import List

from miv.core.operator import Operator, DataLoader
from miv.io.openephys import Data, DataManager
from miv.core.pipeline import Pipeline
from PowerSpectrumAnalysis import PowerSpectrumAnalysis
from miv.datasets.openephys_sample import load_data

save_path = "results/"
path:str = load_data(progbar_disable=True).data_collection_path

dataset: DataManager = DataManager(data_collection_path=path)
data: DataLoader = dataset[0]

bandpower = PowerSpectrumAnalysis(
        channel=[2, 3],
        time=[5, 10],
        frequency_limit=[0, 10],
        win_sec=4,
        band=([0.5, 4], [12, 30]),
        nperseg = 100,
        noverlap= 50,
        band_display = [0, 2],
        db = False,
    )

data >> bandpower
pipeline = Pipeline(bandpower)
pipeline.run(working_directory="results/", verbose=True)

