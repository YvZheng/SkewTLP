# -*- coding: utf-8 -*-
"""压单位统一为hPa，
 温度单位统一为C"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

from libs.VelocityDiagnosis import NanMaInterpHeight, WindInterp, brn, \
    deg2uv, shear, storm_relative_helicity, storm_motion_bunkers
from libs.diagnosis import ReadSoundingData, SoundingInterp, TypeParcel, dry_ascent, \
    Parcel, MutiProfile, cape_cin, Z20, Z0, BCAPE, DCAPE, JI, MJI, SI, SWEAT, BLI, LI, CCL, \
    IC, K_Index, TT_index, TC, precipitable_water, thickness_1000_500

import configs


def physic_cal(filename, parcel_type="sb"):
    dat = ReadSoundingData(filename)  ##读取原始探空资料
    SoundingPres, SoundingHeightM, SoundingTempK, SoundingDwptK, SoundingThetaK, SoundingWind_D, \
    SoundingWind_S = SoundingInterp(dat["Pres_Pa"], dat["HeightM"], dat["TempK"],
                                    dat["DwptK"], dat["Theta"], dat["Wind_D"], dat["Wind_S"])  # 将探空数据间隔规整化
    SoundingU, SoundingV = deg2uv(SoundingWind_D, SoundingWind_S)  # 其中风向、风速转化为u，v

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

    # 计算500hPa、600hPa、700hPa、800hPa高度的环境气温值
    TempK500 = NanMaInterpHeight(50000, SoundingPres, SoundingTempK)
    TempK600 = NanMaInterpHeight(60000, SoundingPres, SoundingTempK)
    TempK700 = NanMaInterpHeight(70000, SoundingPres, SoundingTempK)
    TempK850 = NanMaInterpHeight(85000, SoundingPres, SoundingTempK)

    # 计算500hPa高度的抬升曲线的的温度值
    tparcel500 = NanMaInterpHeight(50000, pparcel, tparcel)

    # 计算500hPa、600hPa、700hPa、800hPa高度的环境露点温度值
    DwptK500 = NanMaInterpHeight(50000, SoundingPres, SoundingDwptK)
    DwptK600 = NanMaInterpHeight(60000, SoundingPres, SoundingDwptK)
    DwptK700 = NanMaInterpHeight(70000, SoundingPres, SoundingDwptK)
    DwptK850 = NanMaInterpHeight(85000, SoundingPres, SoundingDwptK)

    # 计算850hPa,500hPa的风向、风速值
    wind_d850, wind_s850 = WindInterp(85000, SoundingPres, SoundingWind_D, SoundingWind_S)
    wind_d500, wind_s500 = WindInterp(50000, SoundingPres, SoundingWind_D, SoundingWind_S)
    # 计算lfc， el高度，计算cape值、cin值
    plfc, P_el, cape, cin = cape_cin(HeightM_parcel, pparcel, TempKEnv_parcel, tparcel, P_lcl)
    tlfc = NanMaInterpHeight(plfc, SoundingPres, SoundingTempK)
    T_el = NanMaInterpHeight(P_el, SoundingPres, SoundingTempK)

    # 计算-20°C层的气压高度（Pa）、实际高度值（m）
    pa20, zm20 = Z20(SoundingPres, SoundingHeightM, SoundingTempK)
    # 计算0°C层的气压高度（Pa）、实际高度值（m）
    pa0, zm0 = Z0(SoundingPres, SoundingHeightM, SoundingTempK)
    # 计算BCAPE的值
    Bcape = BCAPE(SoundingPres, SoundingTempK, SoundingDwptK, SoundingHeightM)
    # 计算DCAPE的值
    Dcape = DCAPE(TempK600, DwptK600, SoundingPres, SoundingTempK, SoundingDwptK)
    # 计算JI的值
    ji = JI(TempK850, TempK500, DwptK850)
    # 计算MJI的值
    mji = MJI(TempK850, TempK700, TempK500, DwptK850, DwptK700)
    # 计算SI的值
    si = SI(TempK500, TempK850, DwptK850)
    # 计算BLI的值
    bli = BLI(SoundingPres, SoundingTempK, SoundingDwptK)
    # 计算IC的值
    ic = IC(TempK850, TempK500, DwptK850, DwptK500)
    # 计算LI的值
    li = LI(TempK500, tparcel500)
    # 计算K指数的值
    ki = K_Index(TempK850, TempK700, TempK500, DwptK850, DwptK700)
    # 计算TT的值
    tt = TT_index(TempK850, TempK500, DwptK850)
    # 计算SWEAT的值
    sweat = SWEAT(wind_d850, wind_d500, wind_s850, wind_s500, DwptK850, tt)
    # 计算CCL的高度（Pa）
    P_ccl = CCL(T_lcl, P_lcl, SoundingPres, SoundingTempK)
    # 计算CCL高度的温度
    T_ccl = NanMaInterpHeight(P_ccl, SoundingPres, SoundingTempK)
    # 计算对流温度
    tc = TC(P_ccl, T_ccl, pparcel)
    # 计算总可降水量
    tpw = precipitable_water(pparcel, SoundingHeightM, SoundingTempK, SoundingDwptK)
    # 计算BRN的值
    Brn = brn(*deg2uv(SoundingWind_D, SoundingWind_S), SoundingHeightM, cape)
    # 计算1000hPa到500hPa的厚度
    thick = thickness_1000_500(SoundingPres, SoundingHeightM)

    # 计算不同高度的风切变
    shr01 = shear(SoundingU, SoundingV, SoundingHeightM, 0, 1000)
    shr12 = shear(SoundingU, SoundingV, SoundingHeightM, 1000, 2000)
    shr23 = shear(SoundingU, SoundingV, SoundingHeightM, 2000, 3000)
    shr34 = shear(SoundingU, SoundingV, SoundingHeightM, 3000, 4000)
    shr45 = shear(SoundingU, SoundingV, SoundingHeightM, 4000, 5000)
    shr56 = shear(SoundingU, SoundingV, SoundingHeightM, 5000, 6000)

    shr03 = shear(SoundingU, SoundingV, SoundingHeightM, 0, 3000)
    shr06 = shear(SoundingU, SoundingV, SoundingHeightM, 0, 6000)

    # 计算不同高度的srh
    srh01 = storm_relative_helicity(SoundingU, SoundingV, SoundingHeightM, 0, 1000)
    srh03 = storm_relative_helicity(SoundingU, SoundingV, SoundingHeightM, 0, 3000)
    # 计算风暴移动的速度
    storm_u, storm_v, _, _ = storm_motion_bunkers(SoundingU, SoundingV, SoundingHeightM)
    # 计算不同高度的erh
    erh01 = storm_relative_helicity(SoundingU, SoundingV, SoundingHeightM, 0, 1000,
                                    storm_u=storm_u, storm_v=storm_v)

    erh03 = storm_relative_helicity(SoundingU, SoundingV, SoundingHeightM, 0, 3000,
                                    storm_u=storm_u, storm_v=storm_v)

    return dict(zip(["StartPres", "StartTempK", "StartDwptK", "TypeP",
                     "P_lcl", "T_lcl", "TempK500", "TempK600", "TempK700", "TempK850",
                     "DwptK500", "DwptK600", "DwptK700", "DwptK850", "plfc", "tlfc", "P_el", "T_el",
                     "cape", "cin", "pa20", "zm20", "pa0", "zm0", "Bcape", "Dcape", "ji", "mji",
                     "si", "bli", "ic", "li", "ki", "tt", "sweat", "tc", "tpw", "Brn", "thick",
                     "shr01", "shr12", "shr23", "shr34", "shr45", "shr56", "shr03", "shr06",
                     "srh01", "srh03", "erh01", "erh03", ],
                    [StartPres, StartTempK, StartDwptK, TypeP,
                     P_lcl, T_lcl, TempK500, TempK600, TempK700, TempK850,
                     DwptK500, DwptK600, DwptK700, DwptK850, plfc, tlfc, P_el, T_el,
                     cape, cin, pa20, zm20, pa0, zm0, Bcape, Dcape, ji, mji,
                     si, bli, ic, li, ki, tt, sweat, tc, tpw, Brn, thick,
                     shr01, shr12, shr23, shr34, shr45, shr56, shr03, shr06,
                     srh01, srh03, erh01, erh03, ]))


def sys_call(filename=configs.cfgs['SkewT_params']['path_input'],
             parcel_type=configs.cfgs['SkewT_params']['parcel_type']):
    file_save = os.path.splitext(os.path.basename(filename))[0] + "_parameters.json"
    file_full_save = os.path.join(configs.cfgs['SkewT_params']['save_dir'], file_save)
    with open(file_full_save, "w") as f:
        json.dump(physic_cal(filename, parcel_type), f)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("warning using!!! example: sounding_parameter filename, run config path file")
        sys_call()
    elif not os.path.exists(sys.argv[1]):
        print("file is not exist!!!")
    else:
        sys_call(sys.argv[1])
        print("sucessful!!!")
