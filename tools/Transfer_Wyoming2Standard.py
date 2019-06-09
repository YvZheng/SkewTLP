# -*- coding: utf-8 -*-
"""
将怀俄明大学探空数据转换为所需的数据格式
Created on Wed Mar 13 10:15:59 2019

@author: zy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import pandas as pd
from io import StringIO


def transfer(filename):
    skip_rows = 5
    head = ["PRES", "HGHT", "TEMP", "DWPT", "FRPT", "RELH", "RELI", "MIXR", "DRCT", "SKNT", "THTA", "THTE", "THTV"]

    f = open(filename, "r")
    all_text = f.readlines()

    s = StringIO()

    for index, text in enumerate(all_text):
        if text.strip() == "Station information and sounding indices":
            end_index = index
            break
    print(end_index)

    for irows in range(skip_rows, end_index):
        for icols in range(0, 91, 7):
            if (all_text[irows][icols:icols + 7]).isspace() == True:
                all_text[irows] = all_text[irows][:icols] + "    nan" + all_text[irows][icols + 7:]

    s.writelines(all_text)
    s.seek(0, 0)

    dat = pd.read_table(s, delim_whitespace=True, skiprows=skip_rows,
                        nrows=end_index - skip_rows, names=head)

    new_dat = pd.DataFrame(columns=["StationId", "Lat", "Lon", "SoundingDate", "PresHPa",
                                    "HeightM", "TempC", "DwptC", "RH", "Wind_D", "Wind_S"])

    new_dat["PresHPa"] = dat["PRES"]
    new_dat["StationId"] = 72317
    new_dat["Lat"] = 32
    new_dat["Lon"] = 20
    new_dat["SoundingDate"] = 2018060600
    new_dat["HeightM"] = dat["HGHT"]
    new_dat["TempC"] = dat["TEMP"]
    new_dat["DwptC"] = dat["DWPT"]
    new_dat["RH"] = dat["RELH"]
    new_dat["Wind_S"] = dat["SKNT"] * 0.5144
    new_dat["Wind_D"] = dat["DRCT"]

    return new_dat


for filename in glob.glob(r"../data/Wyoming_Sounding/*.txt"):
    new_dat = transfer(filename)
    savename = os.path.splitext(os.path.basename(filename))[0]
    new_dat.to_csv(os.path.join(r"../data/Standard_Input/Standard_skewT_Input/", savename + ".csv"), index=None)
