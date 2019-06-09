# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import warnings
import configs
from libs.diagnosis import ReadSoundingData
from libs.dynamic import sat_mixing_ratio
from libs.VelocityDiagnosis import NanMaInterpHeight
from scipy.interpolate import griddata
import xarray as xr
import json
import os
import sys

def interp5lv(SoundingDat):
    """默认将温度、湿度、风速等插值到给定气压层[200，500，700，850，1000]hpa"""
    interp_pres = np.array(configs.cfgs["Adv_params"]['level'])
    interp_tempk = NanMaInterpHeight(interp_pres, SoundingDat["Pres_Pa"], SoundingDat["TempK"])
    interp_dwptk = NanMaInterpHeight(interp_pres, SoundingDat["Pres_Pa"], SoundingDat["DwptK"])
    sms = SoundingDat['Wind_S']
    drct = SoundingDat['Wind_D']
    rdir = (270. - drct) * (np.pi / 180.)
    uu = np.ma.masked_invalid(sms * np.cos(rdir))
    vv = np.ma.masked_invalid(sms * np.sin(rdir))
    interp_uu = NanMaInterpHeight(interp_pres, SoundingDat["Pres_Pa"], uu)
    interp_vv = NanMaInterpHeight(interp_pres, SoundingDat["Pres_Pa"], vv)
    return interp_pres, interp_tempk, interp_dwptk, interp_uu, interp_vv

def flux_vapour(u, v, q, g=9.8):
    """计算水平水汽通量
    q: 比湿(本站), units:kg/kg
    u: 纬向风速(本站), units: m/s
    v: 经向风速(本站), units: m/s
    g: 重力加速度，units: m/s^2
    return: vapor通量 units: (g/g)*(m/s)/(m/s^2)*1000
    = (g*s)/kg = g/(cm*hpa*s)
    """
    return 1000*u*q/g,1000*v*q/g

def ms2kmh(v):
    """"将m/s转化为km/h"""
    return 3.6*v

def adev_temp(dTdx,dTdy,u,v):
    """计算两站点的温度平流
    dTdx: 纬向温度的变化 c/km
    dTdy: 经向温度的变化 c/km
    u: 本站纬向风速 km/h
    v：本站经向风速 km/h
    return 温度平流units: degree C/hour"""
    return -1*dTdx*u - dTdy*v

def dtds(dt,ds):
    """计算某一方向的梯度,
    假设: t1(C)起始点的温度
    t2(C)终点的温度
    dt = t2 - t1, units : degree C
    假设,s1(km)起点的距离
    s2(km)终点的距离, ds = s2-s1 ,units:km
    """
    return dt/ds

def geographic_to_cartesian(lon, lat, lon_0, lat_0, R=6370997.):
    """
    Azimuthal equidistant geographic to Cartesian coordinate transform.
    Transform a set of geographic coordinates (lat, lon) to
    Cartesian/Cartographic coordinates (x, y) using a azimuthal equidistant
    map projection [1].
    .. math::
        x = R * k * \\cos(lat) * \\sin(lon - lon_0)
        y = R * k * [\\cos(lat_0) * \\sin(lat) -
                        \\sin(lat_0) * \\cos(lat) * \\cos(lon - lon_0)]
        k = c / \\sin(c)
        c = \\arccos(\\sin(lat_0) * \\sin(lat) +
                        \\cos(lat_0) * \\cos(lat) * \\cos(lon - lon_0))
    Where x, y are the Cartesian position from the center of projection;
    lat, lon the corresponding latitude and longitude; lat_0, lon_0 are the
    latitude and longitude of the center of the projection; R is the radius of
    the earth (defaults to ~6371 km).
    Parameters
    ----------
    lon, lat : array-like
    Longitude and latitude coordinates in degrees.
    lon_0, lat_0 : float
    Longitude and latitude, in degrees, of the center of the projection.
    R : float, optional
    Earth radius in the same units as x and y.  The default value is in
    units of meters.
    Returns
    -------
    x, y : array
    Cartesian coordinates in the same units as R, typically meters.
    References
    ----------
    .. [1] Snyder, J. P. Map Projections--A Working Manual. U. S. Geological
    Survey Professional Paper 1395, 1987, pp. 191-202.
    """
    lon = np.atleast_1d(np.asarray(lon))
    lat = np.atleast_1d(np.asarray(lat))

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)

    lon_diff_rad = lon_rad - lon_0_rad

    # calculate the arccos after ensuring all values in valid domain, [-1, 1]
    arg_arccos = (np.sin(lat_0_rad) * np.sin(lat_rad) +
                  np.cos(lat_0_rad) * np.cos(lat_rad) * np.cos(lon_diff_rad))
    arg_arccos[arg_arccos > 1] = 1
    arg_arccos[arg_arccos < -1] = -1
    c = np.arccos(arg_arccos)
    with warnings.catch_warnings():
        # division by zero may occur here but is properly addressed below so
        # the warnings can be ignored
        warnings.simplefilter("ignore", RuntimeWarning)
        k = c / np.sin(c)
    # fix cases where k is undefined (c is zero), k should be 1
    k[c == 0] = 1

    x = R * k * np.cos(lat_rad) * np.sin(lon_diff_rad)
    y = R * k * (np.cos(lat_0_rad) * np.sin(lat_rad) -
                 np.sin(lat_0_rad) * np.cos(lat_rad) * np.cos(lon_diff_rad))
    return x, y

def cal_qflux(filename):
    """计算水汽通量，单位g/(hpa*cm*s)"""
    dat = ReadSoundingData(filename)
    interp_pres, interp_tempk, interp_dwptk, interp_uu, interp_vv = interp5lv(dat)
    interp_q = sat_mixing_ratio(interp_pres, interp_dwptk)
    flux_q_x, flux_q_y = flux_vapour(interp_uu, interp_vv, interp_q)
    return interp_pres, interp_tempk, interp_dwptk, interp_uu, interp_vv, flux_q_x, flux_q_y

def advec_T(data_xarray, dlatdlon=0.5):
    """dlatdlon是指经向纬向的分辨率"""
    lat = data_xarray.lat.values
    lon = data_xarray.lon.values
    tp = data_xarray.tp.values
    uu = data_xarray.u.values
    vv = data_xarray.v.values

    grid_lat, grid_lon = np.mgrid[min(lat):max(lat):dlatdlon, min(lon):max(lon):dlatdlon]

    nstations, nlevel = tp.shape
    ncols, nrows = grid_lat.shape
    grid_tp = np.zeros((nlevel, ncols, nrows))
    grid_uu = np.zeros((nlevel, ncols, nrows))
    grid_vv = np.zeros((nlevel, ncols, nrows))

    dlat = 6378388. * dlatdlon / 180. * np.pi  ##lat方向的梯度的距离,单位m
    dlon = 6378388. * np.cos(
        np.arange(min(lat), max(lat), dlatdlon) / 180. * np.pi) * dlatdlon / 180. * np.pi  ##lon方向的梯度的距离,单位m

    for i in range(nlevel):
        grid_tp[i, :, :] = griddata((lat, lon), tp[:, i], (grid_lat, grid_lon), method='nearest')
        grid_uu[i, :, :] = griddata((lat, lon), uu[:, i], (grid_lat, grid_lon), method='nearest')
        grid_vv[i, :, :] = griddata((lat, lon), vv[:, i], (grid_lat, grid_lon), method='nearest')

    gradient_tp_lat, gradient_tp_lon = np.gradient(grid_tp, axis=(1,2))
    gradient_tp_lat = gradient_tp_lat / dlat
    gradient_tp_lon = gradient_tp_lon / dlon.reshape(1, -1, 1)

    advection_tp = -1 * (gradient_tp_lat * grid_vv + gradient_tp_lon * grid_uu)

    dataset_out = xr.Dataset({'advection_tp': (['level', 'lat', 'lon'],  advection_tp.astype(np.float32)),},
                             coords={'lon': np.arange(min(lon),max(lon),dlatdlon).astype(np.float32),
                                     'lat': np.arange(min(lat),max(lat),dlatdlon).astype(np.float32),
                                     'level': np.array(configs.cfgs["Adv_params"]['level']),
                                    })
    dataset_out.coords["lon"].attrs['units'] = "degrees_east"
    dataset_out.coords["lat"].attrs['units'] = "degrees_north"
    dataset_out.coords['level'].attrs['units'] = "hPa"
    dataset_out.coords["lon"].attrs["long_name"] = "longitude"
    dataset_out.coords["lat"].attrs["long_name"] = "latitude"
    dataset_out.coords["level"].attrs["long_name"] = "Pressure level"

    dataset_out.advection_tp.attrs['units'] = "K/s"
    dataset_out.attrs["time"] = data_xarray.attrs['time']

    return dataset_out

def process_dir(filedir):
    """将同一时间不同站点的探空数据放入同一个文件夹，input该文件夹, output所有站点数据,
        所有站点的纬度, 站点的经度, 站点的插值后不同高度温度（200，500，700，850，1000）,
        不同高度的u风速，不同高度的v风速，不同高度x方向的水汽通量，不同高度y方向水汽通量"""
    lat = []
    lon = []
    tp = []
    uu = []
    vv = []
    flux_q_x = []
    flux_q_y = []
    StationId = []
    for file in os.listdir(filedir):
        filename = filedir + os.sep + file
        interp_pres, interp_tempk, interp_dwptk,\
        interp_uu, interp_vv, iflux_q_x, iflux_q_y = cal_qflux(filename)
        idat = ReadSoundingData(filename)
        lat.append(idat["Lat"][0])
        lon.append(idat["Lon"][0])
        StationId.append(idat['StationId'][0])
        tp.append(interp_tempk)
        uu.append(interp_uu)
        vv.append(interp_vv)
        flux_q_x.append(iflux_q_x)
        flux_q_y.append(iflux_q_y)
    Id = np.asarray(StationId)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    tp = np.asarray(tp)
    uu = np.asarray(uu)
    vv = np.asarray(vv)
    flux_q_x = np.asarray(flux_q_x)
    flux_q_y = np.asarray(flux_q_y)

    StationData = xr.Dataset({'u': (['station', 'level'],  uu.astype(np.float32)),
                              'v': (['station', 'level'],  vv.astype(np.float32)),
                              "flux_q_u": (['station', 'level'],  flux_q_x.astype(np.float32)),
                              "flux_q_v": (['station', 'level'],  flux_q_y.astype(np.float32)),
                              "tp": (['station', 'level'], tp.astype(np.float32)),},
                             coords={'lon': (['station',], lon.astype(np.float32)),
                                     'lat': (['station',], lat.astype(np.float32)),
                                     'level': np.array(configs.cfgs["Adv_params"]['level'], dtype=np.int32),
                                     'station': Id})

    StationData.coords["lon"].attrs['units'] = "degrees_east"
    StationData.coords["lat"].attrs['units'] = "degrees_north"
    StationData.coords['level'].attrs['units'] = "hPa"
    StationData.coords["lon"].attrs["long_name"] = "longitude"
    StationData.coords["lat"].attrs["long_name"] = "latitude"
    StationData.coords["level"].attrs["long_name"] = "Pressure level"

    StationData.u.attrs["units"] = "m/s"
    StationData.v.attrs["units"] = "m/s"

    StationData.flux_q_u.attrs["units"] = "g/(cm*hpa*s)"
    StationData.flux_q_v.attrs["units"] = "g/(cm*hpa*s)"

    StationData.tp.attrs["units"] = "K"
    StationData.tp.attrs["long_name"] = "air temperature"

    StationData.attrs["Description"] = "interped by station data"
    StationData.attrs["time"] = idat['SoundingDate'][0]
    return StationData

def sys_cal(station_dir = configs.cfgs["Adv_params"]["path_input"]):

    flux_q_name = os.path.basename(os.path.normpath(station_dir)) + "_flux_q.json"
    advection_name = os.path.basename(os.path.normpath(station_dir)) + "_advection.json"

    flux_q_path = os.path.join(configs.cfgs["Adv_params"]["save_dir"], flux_q_name)
    advection_path = os.path.join(configs.cfgs["Adv_params"]["save_dir"], advection_name)
    StationData = process_dir(station_dir)
    with open(flux_q_path, 'w') as f:
        json.dump(StationData.to_dict(), f)
    #StationData.to_netcdf(flux_q_path)
    advection_data = advec_T(StationData)
    with open(advection_path, 'w') as f:
        json.dump(advection_data.to_dict(), f)
    #advection_data.to_netcdf(advection_path)
    return


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("warning! example: Advection filedir, now run config path!")
        sys_cal()
    elif not os.path.exists(sys.argv[1]):
        print("file dir is not exist!!!")
    else:
        sys_cal(sys.argv[1])
        print("sucessful!!!")
            
