# -*- coding: utf-8 -*-
"""
诊断常用的函数
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('../')
import numpy as np
from libs.dynamic import degCtoK, Rs_da, Cp_da, VaporPressure, MixRatio, GammaW, \
    Theta, TempK, MixR2VaporPress, DewPoint, RHFromDwpt, DwptkFromRH, \
    PseudoeqPotentialTempkFromDwptk, WetBulbFromDwpt, SpecificVolume, DensHumid
from libs.VelocityDiagnosis import ProcessNan, Monotonic, NanMaInterpPres, NanMaInterpHeight, \
    WindInterp, brn, deg2uv, shear, storm_relative_helicity, storm_motion_bunkers
import pandas as pd
import sys
import os

ms2knot = 1.9438


def ReadSoundingData(filename):
    """
    读取探空数据
    Dat keyword("StationId","Lat","Lon","SoundingDate","PresHPa", "HeightM",
                "TempC", "DwptC", "RH", "Wind_D", "Wind_S","DwptK","RH","Theta","Pres_Pa")
    :param filename:
    :return:dataframe
    """
    fid = open(filename, "r")
    nan_value = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', \
                 '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan', -999, 999999]
    dat = pd.read_csv(fid, na_values=nan_value)
    keywords = ["StationId", "Lat", "Lon", "SoundingDate", "PresHPa", "HeightM",
                "TempC", "DwptC", "RH", "Wind_D", "Wind_S"]
    for ikey in keywords:
        if ikey not in dat.columns:
            raise Exception("file format problems!")
    dat["TempK"] = dat["TempC"] + degCtoK
    dat["Pres_Pa"] = dat["PresHPa"] * 100
    if "DwptC" not in dat.columns:
        dat["DwptK"] = DwptkFromRH(dat["rh"], dat["TempK"])
        dat["DwptC"] = dat["DwptK"] - degCtoK
    else:
        dat["DwptK"] = dat["DwptC"] + degCtoK
    if "RH" not in dat.columns:
        dat["RH"] = RHFromDwpt(dat["TempK"], dat["DwptK"])
    dat["Theta"] = Theta(dat["TempK"], dat["Pres_Pa"])
    return dat


def solve_eq(Pres, TempEnvK, ParcelProfileK):
    """
    计算抬升温度曲线和环境温度曲线的交点
    :param Pres: 气压值（Pa），与抬升温度曲线、环境温度曲线相对应
    :param TempEnvK: 环境温度变化（K）
    :param ParcelProfileK: 抬升温度变化（K）
    :return: 交点的气压值（Pa，list，从大到小排），交点的斜率（与交点的气压值相对应。list）
    """
    Pres, TempEnvK, ParcelProfileK = Monotonic(*ProcessNan(Pres, TempEnvK, ParcelProfileK))
    DiffParcelEnv = ParcelProfileK - TempEnvK
    DiffParcelEnv[DiffParcelEnv == 0] = 1e-5
    dsign = np.sign(DiffParcelEnv)

    EndDiff = np.zeros(dsign.shape, dtype=np.bool)
    EndDiff[1:] = np.abs(np.diff(dsign)).astype(np.bool)  ##符号变化结束位置
    StartDiff = np.zeros(dsign.shape, dtype=np.bool)
    StartDiff[:-1] = EndDiff[1:]  ##符号变化开始位置
    StartDiff[-1] = EndDiff[0]

    sols = np.zeros((EndDiff.sum()))
    stab = np.zeros((EndDiff.sum()))
    for ii in range(EndDiff.sum()):
        xp = np.array([DiffParcelEnv[StartDiff][ii], DiffParcelEnv[EndDiff][ii]])
        fp = np.array([Pres[StartDiff][ii], Pres[EndDiff][ii]])
        sols[ii] = NanMaInterpPres(0, xp, fp)
        # print(DiffParcelEnv[StartDiff][ii], DiffParcelEnv[EndDiff][ii])
        stab[ii] = np.sign(DiffParcelEnv[EndDiff][ii] - DiffParcelEnv[StartDiff][ii])
    return sols[::-1], stab[::-1]


def cape_cin(Height, Pres, TempKEnv, ParcelProfileK, P_lcl, totalcape=False):
    """
    计算对流有效位能CAPE，对流抑制能量CIN，自由对流高度LFC，平衡高度EL
    :param Height:高度值（m），与气压值、环境温度曲线、抬升温度曲线相对应，从小到大变化
    :param Pres: 气压值（Pa），与高度值、环境温度曲线、抬升温度曲线相对应，从大到小变化
    :param TempKEnv:环境温度曲线（K），与气压值相对应、高度值、抬升温度曲线相对应
    :param ParcelProfileK: 抬升温度曲线（K），与高度值、环境温度曲线、气压值相对应
    :param P_lcl: 抬升凝结高度（Pa）
    :param totalcape: 是否计算全部的有效位能和对流抑制位能
    :return:有效位能CAPE(J)，对流抑制能量CIN（J），自由对流高度LFC（Pa），平衡高度EL（Pa）
    """
    assert Pres.shape == ParcelProfileK.shape == TempKEnv.shape == Height.shape

    T_lcl = NanMaInterpHeight(P_lcl, Pres, ParcelProfileK)
    TEnv_lcl = NanMaInterpHeight(P_lcl, Pres, TempKEnv)

    eqPres, Stab = solve_eq(Pres[Pres <= P_lcl], TempKEnv[Pres <= P_lcl], ParcelProfileK[Pres <= P_lcl])

    isstab = np.where(Stab == 1., True, False)
    unstab = np.where(Stab == 1., False, True)

    if eqPres.shape[0] == 0:
        P_lfc = np.nan
        P_el = np.nan
        return P_lfc, P_el, 0, 0  ##lfc, el, cape, cin
    elif T_lcl > TEnv_lcl:
        P_lfc = P_lcl
        if totalcape is True:
            P_el = eqPres[isstab][-1]
        else:
            P_el = eqPres[isstab][0]
    elif eqPres.shape[0] > 1:
        P_lfc = eqPres[unstab][0]
        if totalcape is True:
            P_el = eqPres[isstab][-1]
        else:
            P_el = eqPres[isstab][0]
    else:
        try:
            P_el = eqPres[isstab][0]
            P_lfc = np.nan
        except Exception:
            P_lfc = np.nan
            P_el = np.nan
        return P_lfc, P_el, 0, 0  ##lfc, el, cape, cin

    cape_cond = (ParcelProfileK >= TempKEnv) & (Pres <= P_lfc) & (Pres > P_el)  # 判断cape的条件
    if totalcape is True:  # 判断cin条件
        cin_cond = (ParcelProfileK < TempKEnv) & (Pres > P_el)
    else:
        cin_cond = (ParcelProfileK < TempKEnv) & (Pres > P_lfc)
    CAPE = np.trapz(9.81 * (ParcelProfileK[cape_cond] - TempKEnv[cape_cond]) / TempKEnv[cape_cond],
                    Height[cape_cond])
    CIN = np.trapz(9.81 * (ParcelProfileK[cin_cond] - TempKEnv[cin_cond]) / TempKEnv[cin_cond],
                   Height[cin_cond])
    return P_lfc, P_el, CAPE, CIN


def MutiProfile(pparcel, tparcel, dpparcel, SoundingPres,
                SoundingTempKEnv, SoundingDwptK, SoundingHeight):
    """
    将抬升的温度曲线与环境温度曲线统一, 可用于T-lnP图的作图
    :param pparcel: 抬升过程中气压的变化（Pa）,从大到小 （array）
    :param tparcel: 抬升过程中的温度的变化（K）
    :param dpparcel:抬升过程中的露点温度的变化（K）
    :param SoundingPres:探空的气压值（Pa）， 从大到小
    :param SoundingTempKEnv:探空的温度值、与探空气压值相对应（K）
    :param SoundingDwptK:探空的露点温度值、与探空气压值相对应（K）
    :param SoundingHeight:探空的高度、与探空气压值相对应（m），从小到大
    :return:
    """
    TempKEnv = NanMaInterpHeight(pparcel, SoundingPres, SoundingTempKEnv)
    DwptK = NanMaInterpHeight(pparcel, SoundingPres, SoundingDwptK)
    HeightM = NanMaInterpHeight(pparcel, SoundingPres, SoundingHeight)
    return pparcel, tparcel, dpparcel, TempKEnv, DwptK, HeightM


def TypeParcel(SoundingPres, SoundingHeight, SoundingTempK, SoundingDwptK, SoundingTheta, type="sb"):
    """

    :param SoundingPres:探空的气压值（Pa）， 从大到小
    :param SoundingTempK:探空的温度值、与探空气压值相对应（K）
    :param SoundingDewptK:探空的露点温度值、与探空气压值相对应（K）
    :param type:抬升类型
    :return:
    """
    if type.lower() == "sb":
        return SoundingPres[0], SoundingTempK[0], SoundingDwptK[0], "Surface Base"
    elif type.lower() == "ml":
        depth = 10000
        # identify the layers for averaging
        layers = SoundingPres > SoundingPres[0] - depth
        # average theta over mixheight to give
        thta_mix = SoundingTheta[layers].mean()
        temp_s = TempK(thta_mix, SoundingPres[0])
        # average mixing ratio over mixheight
        vpres = VaporPressure(SoundingDwptK)
        mixr = MixRatio(vpres, SoundingPres)
        mixr_mix = mixr[layers].mean()
        vpres_s = MixR2VaporPress(mixr_mix, SoundingPres[0])
        # surface dew point temp
        dwpt_s = DewPoint(vpres_s)
        return SoundingPres[0], temp_s, dwpt_s, "mixed layer"
    elif type.lower() == "mu":
        depth = 30000
        cape = np.zeros(SoundingPres.shape)
        for ii in range((SoundingPres > SoundingPres[0] - depth).sum()):
            theparcel = SoundingPres[ii], SoundingTempK[ii], SoundingDwptK[ii]
            pparcel, tparcel, dpparcel, P_lcl, T_lcl = Parcel(*theparcel)
            pparcel, tparcel, dpparcel, TempKEnv, DwptK, HeightM = MutiProfile(pparcel,
                                                                               tparcel, dpparcel, SoundingPres,
                                                                               SoundingTempK, SoundingDwptK,
                                                                               SoundingHeight)
            thecape = cape_cin(HeightM, pparcel, TempKEnv, tparcel, P_lcl)[-2]
            cape[ii] = thecape
        if cape.max() == 0:
            return SoundingPres[0], SoundingTempK[0], SoundingDwptK[0], "Most Unstable"
        else:
            I = np.argmax(cape)
            return SoundingPres[I], SoundingTempK[I], SoundingDwptK[I], "Most Unstable"
    else:
        return SoundingPres[0], SoundingTempK[0], SoundingDwptK[0], "Surface Base"


def Parcel(StartPres, StartTempK, StartDewptK):
    """
    计算抬升曲线
    :param StartPres:抬升起始点的气压值（Pa）
    :param StartTempK:抬升起始点的气温值（K）
    :param StartDewptK:抬升起始点的露点温度值（K）
    :return:抬升过程气压变化（Pa，从大到小）, tparcel（抬升过程温度变化）,
     dpparcel（抬升过程的露点温度变化）, P_lcl（抬升凝结高度）, T_lcl（抬升凝结高度的温度）
    """
    PresDry, TempKDry, TempKiso = dry_ascent(
        StartPres, StartTempK, StartDewptK, nsteps=101)
    P_lcl = PresDry[-1]
    T_lcl = TempKDry[-1]
    # print(P_lcl, T_lcl)
    PresWet, TempKWet = moist_ascent(P_lcl, T_lcl, nsteps=201)
    tparcel = np.concatenate((TempKDry, TempKWet[1:]))
    pparcel = np.concatenate((PresDry, PresWet[1:]))  # 气压从大到小
    dpparcel = np.concatenate((TempKiso, TempKWet[1:]))
    return pparcel, tparcel, dpparcel, P_lcl, T_lcl


def moist_descent(StartPres, StartTempK, PresTop=100000, nsteps=201):
    """
    计算湿绝热下沉的气压，气温的变化
    :param StartPres:下沉起始点的气压（Pa）
    :param StartTempK:下沉起始点的气温（K）
    :param PresTop:下沉结束气压（Pa）
    :param nsteps:一共计算的点的个数
    :return:湿绝热下沉的气压变化（Pa, 从小到大），湿绝热下沉的气温的变化（K）
    """
    preswet = np.logspace(np.log10(StartPres), np.log10(PresTop), nsteps)
    TempK = StartTempK
    tempwet = np.zeros(preswet.shape)
    tempwet[0] = StartTempK
    for ii in range(preswet.shape[0] - 1):
        delp = preswet[ii] - preswet[ii + 1]
        TempK = TempK + delp * GammaW(TempK, (preswet[ii] - delp / 2))  ##GammaW湿绝热递减率
        tempwet[ii + 1] = TempK
    return preswet, tempwet


def dry_descent(StartPres, StartTempK, EndPres=100000, nsteps=201):
    """
    计算干绝热下沉的气压，气温的变化
    :param StartPres:下沉起始点的气压（Pa）
    :param StartTempK:下沉起始点的气温（K）
    :param EndPres:下沉结束气压（Pa）
    :param nsteps:一共计算的点的个数
    :return:干绝热下沉的气压变化（Pa, 从小到大），干绝热下沉的气温的变化（K）
    """

    Pres = np.logspace(np.log10(StartPres), np.log10(EndPres), nsteps)
    # Lift the dry parcel
    T_dry = StartTempK * (Pres / StartPres) ** (Rs_da / Cp_da)
    return Pres, T_dry


def moist_ascent(StartPres, StartTempK, PresTop=1000, nsteps=501):
    """
    计算湿绝热过程中的气压的变化，气温的变化，湿绝热过程露点温度==气温！
    :param StartPres:湿绝热抬升起始点的气压（一般为lcl高度，Pa）
    :param StartTempK:湿绝热抬升起始点的气温（一般为lcl的气温，K）
    :param PresTop:抬升的最高点（Pa）
    :param nsteps:抬升中计算的点个数，越密集数据越精细的，但计算速度会变慢
    :return:湿绝热上升的气压变化（Pa，从大到小变化, shape=（nstep，）），
    湿绝热上升的温度变化（K, shape=（nstep，）），
    """
    preswet = np.logspace(np.log10(StartPres), np.log10(PresTop), nsteps)
    tempwet = np.zeros(preswet.shape)
    tempwet[0] = StartTempK
    TempK = StartTempK
    for ii in range(preswet.shape[0] - 1):
        delp = preswet[ii] - preswet[ii + 1]
        TempK = TempK + delp * GammaW(
            TempK, preswet[ii] - delp / 2)  ##GammaW湿绝热递减率
        tempwet[ii + 1] = TempK
    return preswet, tempwet


def dry_ascent(StartPres, StartTempK, StartDewptK, nsteps=101):
    """
    计算干绝热上升中的气温，露点的变化，直至抬升到抬升凝结高度（LCL）
    :param StartPres:抬升起始点的气压（Pa）
    :param StartTempK:抬升起始点的温度（K）
    :param StartDewptK:抬升起始点的露点温度（K），应小于等于温度值
    :param nsteps:抬升中计算的点个数，越密集数据越精细的，但计算速度会变慢
    :return:干绝热上升的气压变化（Pa，从大到小变化, shape=（nstep，）），
    干绝热上升的温度变化（K, shape=（nstep，）），
    干绝热上升的露点温度变化（K, shape=（nstep，））
    """
    assert StartDewptK <= StartTempK

    if StartPres == StartTempK:
        return np.array([StartPres]), np.array([StartDewptK]), np.array([StartTempK])

    # Pres=linspace(StartDewptK,600,nsteps)
    Pres = np.logspace(np.log10(StartPres), np.log10(60000), nsteps)

    # Lift the dry parcel
    T_dry = StartTempK * (Pres / StartPres) ** (Rs_da / Cp_da)

    # Mixing ratio isopleth
    starte = VaporPressure(StartDewptK)  # 根据露点温度的饱和水汽压
    startw = MixRatio(starte, StartPres)  # 计算抬升高度的水汽混合比

    e = Pres * startw / (0.622 + startw)  # 由气压和水汽混合比计算不同高度水汽压
    T_iso = 243.5 / (17.67 / np.log(e / 6.112 / 100.) - 1) + degCtoK  # 由不同高度的水汽压（同一混合比）计算水汽饱和的温度（适用于低温）

    P_lcl = np.interp(0, T_iso - T_dry, Pres)

    presdry = np.logspace(np.log10(StartPres), np.log10(P_lcl), nsteps)
    tempdry = StartTempK * (presdry / StartPres) ** (Rs_da / Cp_da)
    tempiso = 243.5 / (17.67 / np.log((presdry * startw / (0.622 + startw)) / 6.112 / 100.) - 1) \
              + degCtoK

    return presdry, tempdry, tempiso


#####################################################
def SI(TempK500, TempK850, DwptK850):
    """
    计算沙氏指数，850hPa的环境大气进行抬升，抬升到500hPa时，
    环境温度和抬升大气温度的差异
    :param TempK500: 500hPa环境温度
    :param TempK850: 850hPa环境温度
    :param DwptK850: 850hPa环境露点温度
    :return: SI： 沙氏指数
    """
    pparcel, tparcel, _, _, _ = Parcel(85000, TempK850, DwptK850)
    TS500 = NanMaInterpHeight(50000, pparcel, tparcel)
    return TempK500 - TS500


def BLI(Pres, TempK, DwptK):
    """
    最大抬升指数(BLI)是指，把最底层厚度为300hPa
    的大气按50hPa间隔分为许多层，并将各层中间高度处上的
    各点分别按干绝热线上升到各自的抬升凝结高度，然后又分别按
    湿绝热线抬升到500hPa，于是分别得到各点的抬升指数，
    其中正值最大者即为最大抬升指数。
    :param Pres:
    :param TempK:
    :param DwptK:
    :return:
    """
    startlevel = np.linspace(100000, 70000, 7)
    startTempK = NanMaInterpHeight(startlevel, Pres, TempK)
    startDwptK = NanMaInterpHeight(startlevel, Pres, DwptK)
    TempK500 = NanMaInterpHeight(50000, Pres, TempK)
    TS500 = np.zeros_like(startlevel)
    for ii in range(len(startlevel)):
        pparcel, tparcel, _, _, _ = Parcel(startlevel[ii], startTempK[ii], startDwptK[ii])
        TS500[ii] = NanMaInterpHeight(50000, pparcel, tparcel)
    return (TempK500 - TS500).max()


def LI(TempK500, tparcel500):
    """
    计算抬升指数LI，LI=T_500-T_L，其中T_500是指500hPa的实际温度，
    T_L是指气块从自由对流高度开始沿湿绝热线抬升到500hPa的温度；
    其可以定性地用来判断对流层中层（850hPa-500hPa）是否存在热力不稳定层结，
    其与SI指数相似。
    :param TempK500:500hPa环境温度
    :param tparcel500:抬升到500hPa的抬升温度
    :return:Lifted Index
    """
    return TempK500 - tparcel500


def IC(TempK850, TempK500, DwptK850, DwptK500):
    """
    对流稳定的指数；IC=θ_se850-θ_se500，可以用来表征湿空气的条件性静力稳定度，
    若IC大于零，表示层结不稳定，且差值越大越不稳定
    :param TempK850:850hPa环境气温
    :param TempK500:500hPa环境气温
    :return:对流稳定的指数
    """
    Tse850 = PseudoeqPotentialTempkFromDwptk(TempK850, DwptK850, 85000)
    Tse500 = PseudoeqPotentialTempkFromDwptk(TempK500, DwptK500, 50000)
    return Tse850 - Tse500


def K_Index(TempK850, TempK700, TempK500, DwptK850, DwptK700):
    """
    K指数=[T_850-T_500 ]+[T_d ]_850-[T-T_d ]_700，它侧重反应了对流层中
    底层的温湿分布对稳定度的影响
    :param TempK850:850hPa环境气温
    :param TempK700:700hPa环境气温
    :param TempK500:500hPa环境气温
    :param DwptK850:850hPa环境露点
    :param DwptK700:700hPa环境露点
    :return:
    """
    return TempK850 - TempK500 + DwptK850 - (TempK700 - DwptK700) - degCtoK


def TT_index(TempK850, TempK500, DwptK850):
    """
    TT=T_850+T_d850-2T_500，TT越大，越容易发生对流天气。
    :param TempK850:850hPa环境气温
    :param TempK500:500hPa环境气温
    :param DwptK850:850hPa环境露点
    :return:全总指数
    """
    return TempK850 + DwptK850 - 2 * TempK500


def SWEAT(wind_d850, wind_d500, wind_s850, wind_s500, DwptK850, TT):
    """
    通过将几个参数组合成一个指数来评估恶劣天气的可能性。 这些参数包括低空湿度
    （850 mb露点），不稳定性（全总指数），中低水平（850mb和500 mb）风速和暖空气平流
    （850mb至500 mb）。 因此，尝试将运动学和热力学信息合并到一个指数中。 因此，
    SWEAT指数应用于评估恶劣天气潜力，而不是普通的雷暴潜力。
    SWEAT = 12 [Td(850 mb)] + 20 (TT - 49) + 2 (f8) + f5 + 125 (S + 0.2)
    :param wind_d850: 850hPa的风向（degree）
    :param wind_d500: 500hPa的风向（degree）
    :param wind_s850: 850hPa的风速（knot）
    :param wind_s500: 500hPa的风速（knot）
    :param DwptK850: 850hPa的露点
    :param TT: 全总指数
    :return: SWEAT INDEX
    """
    wind_s500 = wind_s500 * ms2knot  # Convert m/s to knot
    wind_s850 = wind_s850 * ms2knot

    if ((wind_d850 <= 250) and (wind_d850 > 130)) and ((wind_d500 <= 310) and (wind_d500 > 210)) and \
            ((wind_d500 - wind_d850) > 0) and ((wind_s500 > 150) and (wind_s500 > 15)):
        s = np.sin((wind_d500 - wind_d850) / 180 * 3.14159)
    else:
        s = -0.2
    if DwptK850 < 0:
        Td850 = 0
    else:
        Td850 = DwptK850 - degCtoK
    if TT > 49:
        SWEAT = 12 * Td850 + 20 * (TT - 49) + 2 * wind_s850 + wind_s500 + 125 * (s + 0.2)
    else:
        SWEAT = 12 * Td850 + 2 * wind_s850 + wind_s500 + 125 * (s + 0.2)
    return SWEAT


def CCL(T_LCL, P_LCL, SoundingPres, SoundingTempK):
    """
    计算自由对流高度
    :param T_LCL: 抬升凝结高度气温（K）
    :param P_LCL: 抬升凝结高度气温（K）
    :param SoundingPres: 探空的气压（Pa）
    :param SoundingTempK:探空的温度（K）
    :return:自由对流高度的气压（Pa）
    """
    mixr = MixRatio(VaporPressure(T_LCL), P_LCL)
    e = MixR2VaporPress(mixr, SoundingPres)
    T_mixratio = 243.5 / (17.67 / np.log(e / 6.112 / 100.) - 1) + degCtoK
    # P_LCL = NanMaInterpPres(0, SoundingTempK - T_mixratio, SoundingPres)
    eqlev, stab = solve_eq(SoundingPres, T_mixratio, SoundingTempK)
    instab = np.where(stab == 1., True, False)
    return eqlev[instab][0]


def TC(P_CCL, T_CCL, SoundingPres):
    """
    计算对流温度TC，气块从对流凝结高度(CCL)沿干绝热
    下降到地表所具有的温度
    :param P_CCL: 对流凝结高度的气压值(Pa)
    :param T_CCL: 对流凝结高度的温度值（K）
    :param SoundingPres:探空的气压变化（Pa，由大到小）
    :return:
    """
    return Theta(T_CCL, P_CCL, SoundingPres[0])


def MJI(TempK850, TempK700, TempK500, DwptK850, DwptK700):
    """
    修正的MJI，  MJI = 1.6*Tw850 - T500 - 0.5*D700 - 8
    :param TempK850:850hPa环境气温
    :param TempK700:700hPa环境气温
    :param Tempk500:500hPa环境气温
    :param DwptK850:850hPa露点温度
    :param DwptK700:700hPa露点温度
    :return:
    """
    wet_bulb850 = WetBulbFromDwpt(TempK850, DwptK850)
    preswet, tempwet = moist_descent(85000, wet_bulb850)
    thetaW = tempwet[-1]
    thetaW = thetaW - degCtoK
    TempK500 = TempK500 - degCtoK
    return 1.6 * thetaW - TempK500 - 0.5 * (TempK700 - DwptK700) - 8


def JI(TempK850, TempK500, DwptK850):
    """
    JI是一种不稳定指数，适用于不同区域及季节，
    其表达式为：T_τ=1.6θ_W850-T_500-11，其中T_τ是修正的雷暴指数，
    θ_W850是850hPa的湿球位温，T_500是500hPa的气温，
    T_τ值指示发生雷暴的可能性
    :param TempK850: 850hPa环境气温
    :param Tempk500: 500hPa环境气温
    :param DwptK850: 850hPa露点温度
    :return:
    """
    wet_bulb850 = WetBulbFromDwpt(TempK850, DwptK850)
    preswet, tempwet = moist_descent(85000, wet_bulb850)
    thetaW = tempwet[-1]
    thetaW = thetaW - degCtoK
    TempK500 = TempK500 - degCtoK
    return 1.6 * thetaW - TempK500 - 11


def DCAPE(TempK600, DwptK600, SoundingPres, SoundingTempK, SoundingDwptK):
    """
    把中层干冷空气的侵入点（可取600hPa）作为下沉起点，下沉起始温度以大气在下沉起点的温度经等焓蒸发
    至饱和时所具有的温度作为大气开始下沉的温度。大气沿假绝热线下沉至大气底，这条假绝
    热线与大气层结曲线所围成的面积所表示的能量为下沉对流有效位能。
    :param TempK600: 600hPa的气温值（K）
    :param DwptK600: 600hPa的露点温度（K）
    :param SoundingPres:探空的气压变化（Pa）
    :param SoundingTempK:探空的温度变化（K）
    :param SoundingDwptK:探空的露点温度变化（K）
    :return:
    """

    TempK600 = NanMaInterpHeight(60000, SoundingPres, SoundingTempK)
    DwptK600 = NanMaInterpHeight(60000, SoundingPres, SoundingDwptK)

    presdry, tempdry, tempiso = dry_ascent(60000, TempK600, DwptK600)
    preswet, tempwet = moist_descent(presdry[-1], tempdry[-1], SoundingPres.max())

    tempenv = NanMaInterpHeight(preswet, SoundingPres, SoundingTempK)
    dtempenv = NanMaInterpHeight(preswet, SoundingPres, SoundingDwptK)

    alpha_env = SpecificVolume(preswet, dtempenv, tempenv)
    alpha_parcel = SpecificVolume(preswet, tempwet, tempwet)
    return np.trapz(alpha_parcel - alpha_env, preswet)


def BCAPE(SoundingPres, SoundingTempK, SoundingDwptK, SoundingHeight):
    """
    在最底层200hPa层次内，找出假相当位温最高值处，将该处气块抬升而算出的CAPE。
    :param SoundingPres:探空的气压变化（Pa）
    :param SoundingTempK:探空的温度变化（K）
    :param SoundingDwptK:探空的露点温度变化（K）
    :param SoundingHeight:探空的高度变化（m）
    :return:
    """
    StartLevel = np.linspace(SoundingPres[0], SoundingPres[0] - 20000, 20)
    TempLevel = NanMaInterpHeight(StartLevel, SoundingPres, SoundingTempK)
    DtLevel = NanMaInterpHeight(StartLevel, SoundingPres, SoundingDwptK)

    EP = PseudoeqPotentialTempkFromDwptk(TempLevel, DtLevel, StartLevel)
    index_max = np.argmax(EP)

    pparcel, tparcel, dpparcel, P_lcl, T_lcl = Parcel(StartLevel[index_max], \
                                                      TempLevel[index_max], DtLevel[index_max])
    pparcel, tparcel, dpparcel, TempKEnv, DwptK, HeightM = \
        MutiProfile(pparcel, tparcel, dpparcel, SoundingPres, SoundingTempK,
                    SoundingDwptK, SoundingHeight)
    _, _, cape, _ = cape_cin(HeightM, pparcel, TempKEnv, tparcel, P_lcl)
    return cape


def Z0(SoundingPres, SoundingHeightM, SoundingTempK):
    """
    计算零度层高度
    :param SoundingPres:探空的气压变化（Pa，由大到小）
    :param SoundingHeightM: 探空高度的变化（m）
    :param SoundingTempK:探空的温度变化（K）
    :return:height(m)
    """
    std_line_0 = np.full_like(SoundingPres, 0 + degCtoK)
    eqPa, Stab = solve_eq(SoundingPres, std_line_0, SoundingTempK)
    if eqPa.shape[0] == 0:
        return np.nan, np.nan
    instab = np.where(Stab == 1., True, False)

    Pa_0 = eqPa[instab][0]
    ZM_0 = NanMaInterpHeight(Pa_0, SoundingPres, SoundingHeightM)
    return Pa_0, ZM_0


def Z20(SoundingPres, SoundingHeightM, SoundingTempK):
    """
    计算-20°C温度层的高度
    :param SoundingPres:探空的气压变化（Pa，由大到小）
    :param SoundingHeightM: 探空高度的变化（m）
    :param SoundingTempK:探空的温度变化（K）
    :return:height(m)
    """
    std_line_neg20 = np.full_like(SoundingPres, -20 + degCtoK)
    eqPa, Stab = solve_eq(SoundingPres, std_line_neg20, SoundingTempK)
    if eqPa.shape[0] == 0:
        return np.nan, np.nan
    # print(eqPa, Stab)
    instab = np.where(Stab == 1., True, False)
    Pa_20 = eqPa[instab][0]
    ZM_20 = NanMaInterpHeight(Pa_20, SoundingPres, SoundingHeightM)
    return Pa_20, ZM_20


def precipitable_water(SoundingPres, SoundingHeightM, SoundingTempK, SoundingDwptK):
    """
    总可降水量mm
    :param SoundingPres:
    :param SoundingHeightM:
    :param SoundingTempK:
    :param SoundingDwptK:
    :return:
    """
    e = VaporPressure(SoundingDwptK)
    mixr = MixRatio(e, SoundingPres)
    rho = DensHumid(SoundingTempK, SoundingPres, e)
    tpw = np.trapz(mixr * rho, SoundingHeightM)
    return tpw


def thickness_1000_500(SoundingPres, SoundingHeightM):
    """
    计算1000hPa到500hPa的厚度
    :param SoundingPres:
    :param SoundingHeightM:
    :return:
    """
    height1000 = NanMaInterpHeight(100000, SoundingPres, SoundingHeightM)
    height500 = NanMaInterpHeight(50000, SoundingPres, SoundingHeightM)
    return height500 - height1000


def SoundingInterp(SoundingPres, SoundingHeightM, SoundingTempK, SoundingDwptK,
                   SoundingThetaK, SoundingWind_D, SoundingWind_S, nsteps=301):
    """
    对探空数据密集化
    :param Pres:
    :param SoundingPres:
    :param SoundingHeightM:
    :param SoundingTempK:
    :param SoundingDwptK:
    :param SoundingWind_D:
    :param SoundingWind_S:
    :return:
    """
    Pres = np.linspace(SoundingPres.max(), SoundingPres.min(), nsteps)
    SoundingHeightM = NanMaInterpHeight(Pres, SoundingPres, SoundingHeightM)
    SoundingTempK = NanMaInterpHeight(Pres, SoundingPres, SoundingTempK)
    SoundingDwptK = NanMaInterpHeight(Pres, SoundingPres, SoundingDwptK)
    SoundingThetaK = NanMaInterpHeight(Pres, SoundingPres, SoundingThetaK)
    SoundingWind_D, SoundingWind_S = WindInterp(Pres, SoundingPres,
                                                SoundingWind_D, SoundingWind_S)
    return Pres, SoundingHeightM, SoundingTempK, SoundingDwptK, SoundingThetaK, \
           SoundingWind_D, SoundingWind_S


def plot_parcel_dat(filename, parcel_type="sb"):
    dat = ReadSoundingData(filename)  ##读取原始探空资料
    SoundingPres, SoundingHeightM, SoundingTempK, SoundingDwptK, SoundingThetaK, SoundingWind_D, \
    SoundingWind_S = SoundingInterp(dat["Pres_Pa"], dat["HeightM"], dat["TempK"],
                                    dat["DwptK"], dat["Theta"], dat["Wind_D"], dat["Wind_S"])  # 将探空数据间隔规整化

    # 按照抬升类型，计算抬升点的气压，温度，露点温度
    StartPres, StartTempK, StartDwptK, TypeP = \
        TypeParcel(SoundingPres, SoundingHeightM, SoundingTempK, SoundingDwptK, SoundingThetaK, parcel_type)

    ##在LCL前的上升过程
    presdry, tempdry, tempiso = dry_ascent(StartPres, StartTempK, StartDwptK)

    # 根据抬升点的气压、气温、露点温度，计算抬升的过程气压、气温、露点变化曲线，同时计算LCL
    pparcel, tparcel, dpparcel, P_lcl, T_lcl = Parcel(StartPres, StartTempK,
                                                      StartDwptK)
    # 将环境气温、露点、高度插值到与抬升的气压变化上，方便cape的计算
    pparcel, tparcel, dpparcel, TempKEnv_parcel, DwptK_parcel, HeightM_parcel = \
        MutiProfile(pparcel, tparcel, dpparcel, SoundingPres, SoundingTempK, SoundingDwptK, SoundingHeightM)

    plfc, P_el, cape, cin = cape_cin(HeightM_parcel, pparcel, TempKEnv_parcel, tparcel, P_lcl)
    tlfc = NanMaInterpHeight(plfc, SoundingPres, SoundingTempK)
    T_el = NanMaInterpHeight(P_el, SoundingPres, SoundingTempK)

    return dict(zip(["pparcel","tparcel", "dpparcel", "TempKEnv_parcel", "DwptK_parcel",
                     "P_lcl", "T_lcl",  "plfc", "tlfc", "P_el", "T_el", "presdry", "tempdry", "tempiso"],
                    [ pparcel,tparcel, dpparcel, TempKEnv_parcel, DwptK_parcel,
                     P_lcl, T_lcl, plfc, tlfc, P_el, T_el, presdry, tempdry, tempiso]))


def splice_key_data(key, data):
    delimiter = ','
    splice_str = [key, ]
    if isinstance(data, np.ndarray) or isinstance(data, tuple):
        str_dat = [str(i) for i in data]
        splice_str.extend(str_dat)
        return delimiter.join(splice_str) + "\n"
    else:
        splice_str.append(str(data))
        print(splice_str)
        return delimiter.join(splice_str) + "\n"


def sys_call(filename, savename, parceltype="ml"):
    text = []
    s = main_cal(filename, parceltype)
    for k in s:
        text.append(splice_key_data(k, s[k]))
    with open(savename, "w") as f:
        f.writelines(text)


#s = main_cal(os.path.abspath(r"./data/Standard_Input/Standard_test.csv"), "mu")
if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("wrong using!!! example: diagnosis filename savename 'ml' ")
    elif not os.path.exists(sys.argv[1]):
        print("file is not exist!!!")
    elif sys.argv[2].lower() not in ["ml", "sb", "mu"]:
        print("pracel type not one of ['ml', 'sb', 'mu']!!!")
        sys_call(sys.argv[1], sys.argv[2])
        print("sucessful!!!")
    else:
        sys_call(sys.argv[1], sys.argv[2], sys.argv[3])
        print("sucessful!!!")
