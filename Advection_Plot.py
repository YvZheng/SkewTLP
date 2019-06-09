# -*- coding: utf-8 -*-
"""多站点联合反演图"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import xarray as xr
import os, sys, json
import configs


def plot_advec(advection_nc="./outputs/advection/20180905231500_advection.json",
               flux_q_nc = "./outputs/advection/20180905231500_flux_q.json",
               level=0):
    """画平流的图在basemap上"""
    #adv = xr.open_dataset(advection_nc)
    #flux = xr.open_dataset(flux_q_nc)
    adv = xr.Dataset.from_dict(json.load(open(advection_nc, "r")))
    flux = xr.Dataset.from_dict(json.load(open(flux_q_nc, "r")))
    max_lat = adv.lat.max()
    min_lat = adv.lat.min()
    max_lon = adv.lon.max()
    min_lon = adv.lon.min()

    lon_0 = (max_lon + min_lon) / 2
    lat_0 = (max_lat + min_lat) / 2

    parallels = np.arange(int(min_lat), int(max_lat) + 1, 0.5)
    meridians = np.arange(int(min_lon), int(max_lon) + 1, 0.5)

    glat_t, glon_t = np.meshgrid(adv.lat, adv.lon)
    grid_lon, grid_lat = glon_t.T, glat_t.T

    fig, ax = plt.subplots()
    m = Basemap(llcrnrlon=min_lon, llcrnrlat=min_lat, urcrnrlon=max_lon, lat_0=lat_0,
                lon_0=lon_0, urcrnrlat=max_lat, projection='lcc', resolution="h", ax=ax)
    x, y = m(grid_lon, grid_lat)
    m.drawcoastlines()
    m.drawcountries()
    pm = m.contourf(x, y, adv.advection_tp[level, :, :], cmap="jet")
    cbar = m.colorbar(pm, location="right", pad="5%")
    cbar.set_label("Temperature advection")
    xq, yq = m(flux.lon.values, flux.lat.values)
    Q = m.quiver(xq, yq, flux.flux_q_u[:, level], flux.flux_q_v[:, level], edgecolor="k", pivot='mid',
                 scale_units='dots', units='dots', scale=0.5, width=2, headwidth=3, headlength=5)
    plt.quiverkey(Q, 0.09, 1.03, 10, r'10 g/(cm*hpa*s)')

    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=6, color="#696969")  # 绘制纬线
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=6, color="#696969")  # 绘制经线
    plt.title(str(flux.level[level].values) + "hPa")

    savename = os.path.join(configs.cfgs["Adv_params"]["save_dir"],
                            os.path.splitext(os.path.basename(advection_nc))[0][:-10] + ".png")
    plt.savefig(savename, ppi=300)
    return

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("warning! example:plot_advec advection_pkl flux_q_pkl level, now run config path!")
        plot_advec()
    elif not os.path.exists(sys.argv[1]):
        print("file is not exist!!!")
    elif not os.path.exists(sys.argv[2]):
        print("file is not exist!!!")
    else:
        plot_advec(sys.argv[1], sys.argv[2], sys.argv[3])
        print("sucessful!!!")