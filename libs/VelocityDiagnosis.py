# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

grav = 9.80665  # Gravity, m s^{-2}
km2m = 1000.


def ProcessNan(*args):
    """处理缺测值，要求输入数据的维度一致，
    剔除np.nan/mask=TRUE的数据

    :param args :(numpy array|mask array)待处理含有缺测的array
    :returns 缺测处理后的数据，维度一致
    """
    mask_array = [np.ma.masked_invalid(i) for i in args]
    mask = np.array([idata.mask for idata in mask_array])
    EffectiveValue = np.all(mask == False, axis=0)
    return [idata[EffectiveValue].data for idata in mask_array]


def Monotonic(*args):
    """对乱序的数据进行整理，使其按照从从小到大排列

    :param args：待排序数据，按照args[0]进行排序
    :returns 排序后的数据
    """
    sort_idx = np.argsort(args[0])
    return [idata[sort_idx] for idata in args]


def uv2deg(u, v):
    """
    将u,v分量转化为风向和风速
    风向是风的来向

    transforms u, v, to direction, maginutide
    :param u: u wind component
    :param v: v wind component
    :returns: wind direction and magnitude
    """
    direction = np.arctan2(u, v) * (180. / np.pi) + 180.
    speed = (u ** 2 + v ** 2) ** (0.5)
    return direction, speed


def deg2uv(direction, speed):
    """
    将风向和风速转化为u，v， 风向是风的来向

    Converts direction and speed into u,v wind

    :param direction: wind direction (mathmatical angle, degree)
    :param speed: wind magnitude (m/s)
    :returns: u and v wind components (m/s)
    """
    u = speed * np.sin(np.pi * (direction + 180.) / 180.)
    v = speed * np.cos(np.pi * (direction + 180.) / 180.)
    return u, v


def NanMaInterp(x, xp, fp):
    """
    用于含缺测数据的插值；
    缺测使用np.nan, mask=True进行表示；
    :param x 待插值高度
    :param xp (array | mask array，1D vector)列表，高度值
    :param fp (array | mask array，1D vector)列表，对应高度的值
    :returns x高度的对应的值
    """
    array_x, array_y = Monotonic(*ProcessNan(xp, fp))
    return np.interp(x, array_x, array_y)


def NanMaInterpHeight(Plev, _pres, _height):
    """
    插值，计算某一气压层的高度值；
    缺测使用np.nan, mask=True进行表示；
    :param Plev 待插值高度的气压
    :param _pres (array | mask array，1D vector)列表，气压值
    :param _height (array | mask array，1D vector)列表，对应高度的值
    :returns Plev高度的对应的高度值
    """
    array_pres, array_height = Monotonic(*ProcessNan(_pres, _height))
    return np.interp(np.log(Plev), np.log(array_pres), array_height)


def NanMaInterpPres(Hlev, _height, _pres):
    """
    插值，计算某一高度的气压值；
    缺测使用np.nan, mask=True进行表示；
    :param Plev 待插值高度的气压
    :param _pres (array | mask array，1D vector)列表，气压值
    :param _height (array | mask array，1D vector)列表，对应高度的值
    :returns Hlev高度的对应的气压值
    """
    array_height, array_pres = Monotonic(*ProcessNan(_height, _pres))
    return np.exp(np.interp(Hlev, array_height, np.log(array_pres)))


def WindInterp(Plev, Pres, direction, speed):
    """
    对风向，风速插值，返回风向、风速
    :param Plev:插值高度
    :param Pres:气压
    :param direction:风向
    :param speed:风速
    :return:插值的结果
    """
    u, v = deg2uv(direction, speed)
    u_plev = NanMaInterpHeight(Plev, Pres, u)
    v_plev = NanMaInterpHeight(Plev, Pres, v)
    return uv2deg(u_plev, v_plev)


def shear(_u, _v, _z, zbot, ztop):
    """
    计算ztop，zbot之间的风切变；
    Calculates the shear in the layer between zbot and ztop
    :param _u: U winds (1D vector in z)
    :param _u: V winds (1D vector in z)
    :param _z: z heights (1D vector in z, increasing)
    :param zbot: Bottom of the layer
    :param ztop: Top of the layer
    """
    ubot = NanMaInterp(zbot, _z, _u)
    vbot = NanMaInterp(zbot, _z, _v)
    utop = NanMaInterp(ztop, _z, _u)
    vtop = NanMaInterp(ztop, _z, _v)
    u = utop - ubot
    v = vtop - vbot
    return u, v


def mean_wind(_u, _v, _z, zbot, ztop):
    """
    计算zbot,ztop之间的平均风速
    Calculates the mean wind in the layer between zbot and ztop
    :param _u: U winds (1D vector in z， m/s)
    :param _u: V winds (1D vector in z, m/s)
    :param _z: z heights (1D vector in z, m, increasing)
    :param zbot: Bottom of the layer(m)
    :param ztop: Top of the layer(m)
    """
    if zbot < _z[0]:
        zbot = _z[0]
    dz = 10.
    z = np.arange(zbot, ztop + dz, dz)
    u = NanMaInterp(z, _z, _u)
    v = NanMaInterp(z, _z, _v)
    uavg = np.mean(u, dtype=np.float64)
    vavg = np.mean(v, dtype=np.float64)

    return uavg, vavg


def brn(_u, _v, _z, cape):
    """
    计算粗理查逊数BRN
    :param _u: U winds (1D vector in z， m/s)
    :param _u: V winds (1D vector in z, m/s)
    :param _z: z heights (1D vector in z, m, increasing)
    :param cape: cape值（J）
    """
    u06avg = mean_wind(_u, _v, _z, 0., 6000.)  # 0-6000m平均风速
    u0500avg = mean_wind(_u, _v, _z, 0., 500.)  # 0-500m平均风速
    u = u06avg[0] - u0500avg[0]  # delta U风速切变值
    v = u06avg[1] - u0500avg[1]  # delta V风速切变值
    brn = cape / (0.5 * (u ** 2 + v ** 2))
    return brn


def storm_relative_helicity(_u, _v, _z, zbot, ztop, storm_u=0, storm_v=0):
    # Partially adapted from similar SharpPy code
    r"""
    计算zbot与ztop之间的风暴相对螺旋度， storm_u， storm_v系统移动速度；
    Calculate storm relative helicity.

    Calculates storm relatively helicity following [Markowski2010] 230-231.

    .. math:: \int\limits_0^d (\bar v - c) \cdot \bar\omega_{h} \,dz

    This is applied to the data from a hodograph with the following summation:

    .. math:: \sum_{n = 1}^{N-1} [(u_{n+1} - c_{x})(v_{n} - c_{y}) -
                                      (u_{n} - c_{x})(v_{n+1} - c_{y})]
    Parameters
    ----------
    _u : array-like
        u component winds
    _v : array-like
        v component winds
    _z : array-like
        atmospheric heights, will be converted to AGL
    zbot : m
        height of layer top AGL
    ztop : m
        height of layer bottom AGL (default is surface)
    storm_u : number
        u component of storm motion (default is 0 m/s)
    storm_v : number
        v component of storm motion (default is 0 m/s)

    Returns
    -------
        positive, negative, total storm-relative helicity
    """
    if zbot < _z[0]:
        zbot = _z[0]
    dz = 10.
    z = np.arange(zbot, ztop + dz, dz)
    u = NanMaInterp(z, _z, _u)
    v = NanMaInterp(z, _z, _v)

    storm_relative_u = u - storm_u
    storm_relative_v = v - storm_v

    int_layers = (storm_relative_u[1:] * storm_relative_v[:-1] -
                  storm_relative_u[:-1] * storm_relative_v[1:])

    # Need to manually check for masked value because sum() on masked array with non-default
    # mask will return a masked value rather than 0. See numpy/numpy#11736
    positive_srh = int_layers[int_layers > 0.].sum()
    #        if np.ma.is_masked(positive_srh):
    #            positive_srh = 0.0
    negative_srh = int_layers[int_layers < 0.].sum()
    #        if np.ma.is_masked(negative_srh):
    #            negative_srh = 0.0
    return positive_srh, negative_srh, (positive_srh + negative_srh)


def storm_motion_rasmussen(_u, _v, _z):
    """calculate storm motion, rasmussen (1984)
    :param _u: U winds (1D vector in z)
    :param _u: V winds (1D vector in z)
    :param _z: z heights (1D vector in z)
    :param u_cr :风暴移动速度u
    :param v_cr :风暴移动速度v
    """
    u_0_500 = mean_wind(_u, _v, _z, 0., 500.)
    u_4km = mean_wind(_u, _v, _z, 3500., 4500.)
    dist_60pct = np.sqrt((u_4km[0] - u_0_500[0]) ** 2 + (u_4km[1] - u_0_500[1]) ** 2) * 0.6
    du = u_4km[0] - u_0_500[0]
    dv = u_4km[1] - u_0_500[1]
    theta = np.arctan(dv / du)
    u60 = dist_60pct * np.cos(theta) + u_0_500[0]
    v60 = dist_60pct * np.sin(theta) + u_0_500[1]
    theta -= np.pi / 2.
    dist = 8.7
    u_cr = dist * np.cos(theta) + u60
    v_cr = dist * np.sin(theta) + v60
    theta += np.pi
    u_cl = dist * np.cos(theta) + u60
    v_cl = dist * np.sin(theta) + v60
    return u_cr, v_cr, u_cl, v_cl


def storm_motion_bunkers(_u, _v, _z):
    """calculate storm motion, bunkers (2000)
    :param _u: U winds (1D vector in z)
    :param _u: V winds (1D vector in z)
    :param _z: z heights (1D vector in z)
    :param u_cr :风暴移动速度u
    :param v_cr :风暴移动速度v
    """
    u_0_500 = mean_wind(_u, _v, _z, 0., 500.)
    u_0_6km = mean_wind(_u, _v, _z, 0., 6000.)
    du = u_0_6km[0] - u_0_500[0]
    dv = u_0_6km[1] - u_0_500[1]
    theta = np.arctan(dv / du) - np.pi / 2.
    dist = 7.5
    u_cr = dist * np.cos(theta) + u_0_6km[0]
    v_cr = dist * np.sin(theta) + u_0_6km[1]
    theta += np.pi
    u_cl = dist * np.cos(theta) + u_0_6km[0]
    v_cl = dist * np.sin(theta) + u_0_6km[1]
    return u_cr, v_cr, u_cl, v_cl
