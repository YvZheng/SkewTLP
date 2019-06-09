# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import sys

sys.path.append("../")

import pandas as pd
from libs.dynamic import degCtoK, DwptkFromRH

file_loc = r"../data/Old_input/*.csv"

for filename in glob.glob(file_loc):
    dat = pd.read_csv(filename, skiprows=1)

    new_dat = pd.DataFrame(columns=["StationId", "Lat", "Lon",
                                    "SoundingDate", "PresHPa", "HeightM", "TempC", "DwptC",
                                    "RH", "Wind_D", "Wind_S"])
    new_dat["StationId"] = dat["StationId"]
    new_dat["Lat"] = dat["Lat"]
    new_dat["Lon"] = dat["Lon"]
    new_dat["SoundingDate"] = dat["SoundingDate"]
    new_dat["PresHPa"] = dat["PRS_HWC"]
    new_dat["HeightM"] = dat["GPH"]
    new_dat["TempC"] = dat["TEM"]
    new_dat["DwptC"] = DwptkFromRH(dat["RH"], dat["TEM"] + degCtoK) - degCtoK
    new_dat["RH"] = dat["RH"]
    new_dat["Wind_S"] = dat["WIN_S"]
    new_dat["Wind_D"] = dat["WIN_D"]

    savename = os.path.basename(filename)
    new_dat.to_csv(os.path.join(r"../data/Standard_Input/Standard_skewT_Input", savename), index=None)
