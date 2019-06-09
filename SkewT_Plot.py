# -*- coding: utf-8 -*-
"""压单位统一为hPa，
 温度单位统一为C"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import numpy as np
import configs

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
from matplotlib.projections import register_projection
from matplotlib.ticker import (MultipleLocator, ScalarFormatter)
from libs.dynamic import degCtoK, Rs_da, Cp_da, barometric_equation_inv, GammaW
from collections import UserDict
from libs.diagnosis import ReadSoundingData, plot_parcel_dat
import sys, os


def moist_as(startp, startt, ptop=10, nsteps=501):
    # -------------------------------------------------------------------
    # Lift a parcel moist adiabatically from startp to endp.
    # Init temp is startt in C, pressure levels are in hPa
    # -------------------------------------------------------------------

    # preswet=linspace(startp,ptop,nsteps)
    preswet = np.logspace(np.log10(startp), np.log10(ptop), nsteps)
    temp = startt
    tempwet = np.zeros(preswet.shape)
    tempwet[0] = startt
    for ii in range(preswet.shape[0] - 1):
        delp = preswet[ii] - preswet[ii + 1]
        temp = temp + 100 * delp * GammaW(
            temp + degCtoK, (preswet[ii] - delp / 2) * 100)  ##GammaW湿绝热递减率
        tempwet[ii + 1] = temp

    return preswet, tempwet


class SkewXTick(maxis.XTick):
    """
    The sole purpose of this class is to look at the upper, lower, or total
    interval as appropriate and see what parts of the tick to draw, if any.
    画图过程中气压单位统一为hPa， 温度单位统一为C，方便可视化。
    """

    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__name__)

        lower_interval = self.axes.xaxis.lower_interval
        upper_interval = self.axes.xaxis.upper_interval

        if self.gridOn and transforms.interval_contains(
                self.axes.xaxis.get_view_interval(), self.get_loc()):
            self.gridline.draw(renderer)

        if transforms.interval_contains(lower_interval, self.get_loc()):
            if self.tick1On:
                self.tick1line.draw(renderer)
            if self.label1On:
                self.label1.draw(renderer)

        if transforms.interval_contains(upper_interval, self.get_loc()):
            if self.tick2On:
                self.tick2line.draw(renderer)
            if self.label2On:
                self.label2.draw(renderer)

        renderer.close_group(self.__name__)


class SkewXAxis(maxis.XAxis):
    """
    This class exists to provide two separate sets of intervals to the tick,
    as well as create instances of the custom tick
    """

    def __init__(self, *args, **kwargs):
        maxis.XAxis.__init__(self, *args, **kwargs)
        self.upper_interval = 0.0, 1.0

    def _get_tick(self, major):
        return SkewXTick(self.axes, 0, '', major=major)

    @property
    def lower_interval(self):
        return self.axes.viewLim.intervalx

    def get_view_interval(self):
        return self.upper_interval[0], self.axes.viewLim.intervalx[1]


class SkewSpine(mspines.Spine):
    """
    This class exists to calculate the separate data range of the
    upper X-axis and draw the spine there. It also provides this range
    to the X-axis artist for ticking and gridlines
    """

    def _adjust_location(self):
        trans = self.axes.transDataToAxes.inverted()
        if self.spine_type == 'top':
            yloc = 1.0
        else:
            yloc = 0.0
        left = trans.transform_point((0.0, yloc))[0]
        right = trans.transform_point((1.0, yloc))[0]

        pts = self._path.vertices
        pts[0, 0] = left
        pts[1, 0] = right
        self.axis.upper_interval = (left, right)


class SkewXAxes(Axes):
    """
    This class handles registration of the skew-xaxes as a projection as well
    as setting up the appropriate transformations. It also overrides standard
    spines and axes instances as appropriate.
    The projection must specify a name.  This will be used be the
    user to select the projection, i.e. ``subplot(111,
    projection='skewx')``.
    """
    name = 'skewx'

    def __init__(self, *args, **kwargs):
        r"""Initialize `SkewXAxes`.

        Parameters
        ----------
        args : Arbitrary positional arguments
            Passed to :class:`matplotlib.axes.Axes`

        position: int, optional
            The rotation of the x-axis against the y-axis, in degrees.

        kwargs : Arbitrary keyword arguments
            Passed to :class:`matplotlib.axes.Axes`

        """
        # This needs to be popped and set before moving on
        self.rot = kwargs.pop('rotation', 45)
        Axes.__init__(self, *args, **kwargs)

    def _init_axis(self):
        # Taken from Axes and modified to use our modified X-axis
        self.xaxis = SkewXAxis(self)
        self.spines['top'].register_axis(self.xaxis)
        self.spines['bottom'].register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self)
        self.spines['left'].register_axis(self.yaxis)
        self.spines['right'].register_axis(self.yaxis)

    def _gen_axes_spines(self):
        spines = {'top': SkewSpine.linear_spine(self, 'top'),
                  'bottom': mspines.Spine.linear_spine(self, 'bottom'),
                  'left': mspines.Spine.linear_spine(self, 'left'),
                  'right': mspines.Spine.linear_spine(self, 'right')}
        return spines

    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """

        # Get the standard transform setup from the Axes base class
        Axes._set_lim_and_transforms(self)

        # Need to put the skew in the middle, after the scale and limits,
        # but before the transAxes. This way, the skew is done in Axes
        # coordinates thus performing the transform around the proper origin
        # We keep the pre-transAxes transform around for other users, like the
        # spines for finding bounds
        self.transDataToAxes = self.transScale + (
                self.transLimits + transforms.Affine2D().skew_deg(self.rot, 0))

        # Create the full transform from Data to Pixels
        self.transData = self.transDataToAxes + self.transAxes

        # Blended transforms like this need to have the skewing applied using
        # both axes, in axes coords like before.
        self._xaxis_transform = (transforms.blended_transform_factory(
            self.transScale + self.transLimits,
            transforms.IdentityTransform()) +
                                 transforms.Affine2D().skew_deg(self.rot, 0)) + self.transAxes


class SkewXAxes(SkewXAxes):
    # In the SkewT package, SkewXAxes is a subclass of the one provided
    # from the example on his webpage (circa 2011) or
    # the example on the matplotlib page. I add the following methods.

    def other_housekeeping(self, mixratio=np.array([])):
        # Added by Thomas Chubb
        self.yaxis.grid(True, ls='-', color='y', lw=0.5)
        majorLocatorDegC = MultipleLocator(10)

        self.xaxis.set_major_locator(majorLocatorDegC)
        self.xaxis.grid(True, color='y', lw=0.5, ls='-')

        # self.set_ylabel('Pressure (hPa)')
        self.set_xlabel('气温: (摄氏度)', fontproperties="SimHei")
        self.set_ylabel('气压: (百帕)', fontproperties="SimHei")
        # explicitly set Y-coord as otherwise "posx and posy should
        # be finite values" error occurs
        self.xaxis.set_label_coords(0.5, -0.05)
        yticks = np.linspace(100, 1000, 10)
        if self.pmin < 100:
            yticks = np.concatenate((np.array([10, 20, 50]), yticks))

        self.set_yticks(yticks)

        self.yaxis.set_major_formatter(ScalarFormatter())
        self.set_xlim(self.tmin, self.tmax)
        self.set_ylim(self.pmax, self.pmin)
        self.spines['right'].set_visible(False)
        self.get_yaxis().set_tick_params(which="both", size=0)
        self.get_xaxis().set_tick_params(which="both", size=0)

    def add_dry_adiabats(self, T0, P, do_labels=True, **kwargs):
        # Added by Thomas Chubb
        P0 = 1000.
        T = np.array([(st + degCtoK) * (P / P0) ** (Rs_da / Cp_da) - degCtoK for st in T0])
        labelt = [(st + degCtoK) * 1 ** (Rs_da / Cp_da) for st in T0]

        # gets a pressure level about 1/4 the way up the plot...
        pp = 10 ** (np.log10(self.pmin ** .2 * self.pmax ** .8))
        xi = np.where(np.abs(P - pp) - np.abs(P - pp).min() < 1e-6)[0][0]

        ndec = np.log10(self.pmax / pp) / np.log10(self.pmax / self.pmin)
        tran = self.tmax - self.tmin
        tminl = self.tmin - tran * ndec
        tmaxl = self.tmax - tran * ndec

        if 'color' in kwargs:
            col = kwargs['color']
        else:
            col = 'k'
        for tt, ll in zip(T, labelt):
            self.plot(tt, P, **kwargs)
            if do_labels:
                if tt[xi] > tmaxl - 2:
                    continue
                if tt[xi] < tminl + 2:
                    continue
                self.text(tt[xi], P[xi] + 10, '%d' % (ll), fontsize=8,
                          ha='center', va='bottom', rotation=-30, color=col,
                          bbox={'facecolor': 'w', 'edgecolor': 'w'})
        return T

    def add_moist_adiabats(self, T0, P0, tmaxl=None, do_labels=True, **kwargs):
        moist_adiabats = np.array([moist_as(P0, st) for st in T0])
        T = moist_adiabats[:, 1, :]
        P = moist_adiabats[0, 0, :]

        # gets a pressure level about 3/4 the way up the plot...
        pp = 10 ** (np.log10(self.pmin ** .75 * self.pmax ** .25))
        xi = np.where(abs(P - pp) - abs(P - pp).min() < 1e-6)[0][0]

        ndec = np.log10(self.pmax / pp) / np.log10(self.pmax / self.pmin)

        tran = self.tmax - self.tmin
        tminl = self.tmin - tran * ndec
        if tmaxl is None:
            tmaxl = self.tmax - tran * ndec

        if 'color' in kwargs:
            col = kwargs['color']
        else:
            col = 'k'
        for tt in T:
            self.plot(tt, P, **kwargs)
            # if (tt[-1]>-60) and (tt[-1]<-10):
            if do_labels:
                if tt[xi] > tmaxl - 2:
                    continue
                if tt[xi] < tminl + 2:
                    continue
                self.text(
                    tt[xi], P[xi], '%d' % tt[0], ha='center', va='bottom',
                    fontsize=8,
                    bbox={'facecolor': 'w', 'edgecolor': 'w'}, color=col)

    def add_mixratio_isopleths(self, w, P, do_labels=True, **kwargs):
        e = np.array([P * ww / (.622 + ww) for ww in w])
        T = 243.5 / (17.67 / np.log(e / 6.112) - 1)
        if 'color' in kwargs:
            col = kwargs['color']
        else:
            col = 'k'

        pp = 700.
        xi = np.where(abs(P - pp) - abs(P - pp).min() < 1e-6)[0][0]

        ndec = np.log10(self.pmax / pp) / np.log10(self.pmax / self.pmin)
        tran = self.tmax - self.tmin
        tminl = self.tmin - tran * ndec
        tmaxl = self.tmax - tran * ndec

        for tt, mr in zip(T, w):
            self.plot(tt, P.flatten(), **kwargs)
            if do_labels:
                if tt[xi] > tmaxl - 2:
                    continue
                if tt[xi] < tminl + 2:
                    continue
                if mr * 1000 < 0.1:
                    fmt = "%4.2f"
                elif mr * 1000 <= 1.:
                    fmt = "%4.1f"
                else:
                    fmt = "%d"
                self.text(
                    tt[-1], P[-1], fmt % (mr * 1000), color=col, fontsize=8,
                    ha='center', va='bottom',
                    bbox={'facecolor': 'w', 'edgecolor': 'w'})


# Now register the projection with matplotlib so the user can select
# it.
register_projection(SkewXAxes)


class Sounding(UserDict):

    def __init__(self, SoundingData):
        self.soundingdata = SoundingData

    def make_skewt_axes(self, pmax=1050., pmin=100., tmin=-40., tmax=30.,
                        fig=None, rotation=45):
        """Set up the skew-t axis """
        font = {'family': 'SimHei',
                'color': 'purple',
                'weight': 'normal',
                'size': 12, }
        if rotation == 0:
            tmin = -80

        if fig is None:
            self.fig = plt.figure(figsize=(8, 8))
            # self.fig.clf()
        else:
            self.fig = fig

        plt.rcParams.update({'font.size': 10, })

        self.skewxaxis = self.fig.add_axes([.1, .1, .8, .8],
                                           projection='skewx', rotation=rotation)
        self.skewxaxis.set_yscale('log')
        self.skewxaxis.pmax = pmax
        self.skewxaxis.pmin = pmin
        self.skewxaxis.tmax = tmax
        self.skewxaxis.tmin = tmin

        xticklocs = np.arange(-80, 45, 10)
        T0 = xticklocs

        P = np.logspace(np.log10(pmax), np.log10(pmin), 101)

        if rotation == 0:
            w = np.array([0.00001, 0.0001, 0.0004, 0.001, 0.002, 0.004, 0.01])
            self.skewxaxis.add_mixratio_isopleths(
                w, P[P >= 700], color='g', ls='--', alpha=0.9, lw=0.5)
            self.skewxaxis.add_dry_adiabats(
                np.linspace(230, 550, 17) - degCtoK, P, color='g', ls='--', alpha=.7,
                lw=0.5)
            self.skewxaxis.add_moist_adiabats(
                np.linspace(21, 51, 6), pmax, tmaxl=45, color='g', ls='--', alpha=.5,
                lw=0.5)
        else:
            w = np.array([0.00001, 0.0001, 0.0004, 0.001, 0.002, 0.004,
                          0.007, 0.01, 0.016, 0.024, 0.032])
            self.skewxaxis.add_mixratio_isopleths(
                w, P[P >= 700], color='g', ls='--', alpha=0.9, lw=0.5)
            self.skewxaxis.add_dry_adiabats(
                np.linspace(210, 550, 18) - degCtoK, P, color='g', ls='--', alpha=.7,
                lw=0.5)
            self.skewxaxis.add_moist_adiabats(
                np.linspace(0, 44, 12), pmax, color='g', ls='--', alpha=.5, lw=0.5)

        #        self.skewxaxis.set_title("%s %s" % (self['StationNumber'],
        #                                            self['SoundingDate']))
        #        self.skewxaxis.set_title(u"Station: NanJing  Date: 20180705 00:00:00")

        self.skewxaxis.other_housekeeping()

        self.wbax = self.fig.add_axes([0.81, 0.1, 0.1, 0.8],
                                      sharey=self.skewxaxis, frameon=False)
        self.wbax.xaxis.set_ticks([], [])
        self.wbax.yaxis.grid(True, ls='-', color='y', lw=0.5)
        for tick in self.wbax.yaxis.get_major_ticks():
            # tick.label1On = False
            pass
        self.wbax.get_yaxis().set_tick_params(size=0, color='y')
        self.wbax.set_xlim(-1.5, 1.5)
        self.wbax.get_yaxis().set_visible(False)
        self.wbax.set_title('m/s', fontsize=10, color='k', ha='right')

        # Set up standard atmosphere height scale on
        # LHS of plot.
        majorLocatorKM = MultipleLocator(2)
        #        majorLocatorKFT = MultipleLocator(5)
        minorLocator = MultipleLocator(1)

        # determine base height from base pressure (nominally 1050 hPa)
        # via hydrostatic equilibrium for standard atmosphere

        # model atmospheric conditions with constant lapse rate and
        # NIST (1013.25hPa and 20C)
        zmin = barometric_equation_inv(0, 293.15, 101325., pmax * 100.)
        zmax = barometric_equation_inv(0, 293.15, 101325., pmin * 100.)

        self.kmhax = self.fig.add_axes([0.9, 0.1, 1e-6, 0.8], frameon=True)
        self.kmhax.xaxis.set_ticks([], [])
        self.kmhax.spines['left'].set_color('k')
        self.kmhax.spines['right'].set_visible(False)
        self.kmhax.tick_params(axis='y', colors='k', labelsize=10, labelright=True, labelleft=False)
        self.kmhax.set_ylim(zmin * 1e-3, zmax * 1e-3)
        self.kmhax.set_title("km", fontsize=10)
        self.kmhax.get_yaxis().set_tick_params(which="both", direction='in')
        self.kmhax.yaxis.set_major_locator(majorLocatorKM)
        self.kmhax.yaxis.set_minor_locator(minorLocator)

    def add_profile(self, **kwargs):
        """Add a new profile to the SkewT plot.

        This is abstracted from plot_skewt to enable the plotting of
        multiple profiles on a single axis, by updating the data attribute.
        For example:
        >>>
        dat = ReadSoundingData(filename)
        plot = Sounding(dat)
        plot.make_skewt_axes()
        plot.add_profile()
        >>>
        Use the kwarg 'bloc' to set the alignment of the wind barbs from
        the centerline (useful if plotting multiple profiles on the one axis)
        >>>
        Modified 25/07/2013: enforce masking of input soundingdata for this
        function (does not affect the data attribute)."""

        if 'bloc' in kwargs:
            bloc = kwargs.pop('bloc')
        else:
            bloc = 0.5
        try:
            pres = np.ma.masked_invalid(self.soundingdata['PresHPa'])
        except KeyError:
            raise KeyError("Pres in hPa (PRES) is required!")

        try:
            tc = np.ma.masked_invalid(self.soundingdata['TempC'])
        except KeyError:
            raise KeyError("Temperature in C (TEMP) is required!")

        try:
            dwpt = np.ma.masked_invalid(self.soundingdata['DwptC'])
        except KeyError:
            print("Warning: No DWPT available")
            dwpt = np.ma.masked_array(np.zeros(pres.shape), mask=True)

        try:
            sms = self.soundingdata['Wind_S']
            drct = self.soundingdata['Wind_D']
            rdir = (270. - drct) * (np.pi / 180.)
            uu = np.ma.masked_invalid(sms * np.cos(rdir))
            vv = np.ma.masked_invalid(sms * np.sin(rdir))
        except KeyError:
            print("Warning: No SMS/DRCT available")
            uu = np.ma.masked_array(np.zeros(pres.shape), mask=True)
            vv = np.ma.masked_array(np.zeros(pres.shape), mask=True)

        tcprof = self.skewxaxis.plot(tc, pres, zorder=5, label="环境温度曲线", **kwargs)
        dpprof = self.skewxaxis.plot(dwpt, pres, zorder=5, ls='--', label="露点温度曲线", **kwargs)

        # this line should no longer cause an exception
        nbarbs = (~uu.mask).sum()

        skip = max(1, int(nbarbs // 32))

        if 'color' in kwargs:
            bcol = kwargs['color']
        else:
            bcol = 'k'

        if 'alpha' in kwargs:
            balph = kwargs['alpha']
        else:
            balph = 1.

        self.wbax.barbs((np.zeros(pres.shape) + bloc)[::skip] - 0.5, pres[::skip],
                        uu[::skip], vv[::skip],
                        length=6, barb_increments=dict(half=2, full=4, flag=20),
                        color=bcol, alpha=balph, lw=0.5)
        self.skewxaxis.other_housekeeping()
        return tcprof

    def lift_parcel(self, dat_dict, *args, **kwargs):
        """Do a lifted parcel analysis on the sounding data"""

        if 'totalcape' in kwargs:
            totalcape = kwargs.pop('totalcape')
        else:
            totalcape = False

        # zorder
        zo = 4
        # trace colour
        col = [.6, .6, .6]

        # Plot traces below LCL
        self.skewxaxis.plot(dat_dict["tparcel"] - degCtoK, dat_dict["pparcel"] / 100.,
                            color=col, lw=2, zorder=zo, label="抬升上升曲线")
        self.skewxaxis.plot(dat_dict["T_lcl"] - degCtoK, dat_dict["P_lcl"] / 100., ls='', marker='o', mec=col,
                            mfc=col, zorder=zo)
        # Plot trace above LCL
        self.skewxaxis.plot(dat_dict["tempiso"] - degCtoK, dat_dict["presdry"] / 100.,
                            color=col, lw=2, zorder=zo, )
        # Plot LFC and EL
        self.skewxaxis.plot(dat_dict["tlfc"] - degCtoK, dat_dict["plfc"] / 100., ls='', marker='o', mew=2, mec='b',
                            mfc='None', zorder=zo)
        self.skewxaxis.plot(dat_dict["T_el"] - degCtoK, dat_dict["P_el"] / 100., ls='', marker='o', mew=2, mec='r',
                            mfc='None', zorder=zo)

        if not np.isnan(dat_dict["plfc"]):
            # Hatch areas of POSITIVE Bouyancy
            cond1 = (dat_dict["tparcel"] >= dat_dict["TempKEnv_parcel"]) & \
                    (dat_dict["pparcel"] <= dat_dict["plfc"]) & (dat_dict["pparcel"] > dat_dict["P_el"])
            self.skewxaxis.fill_betweenx(
                dat_dict["pparcel"] / 100, dat_dict["tparcel"] - degCtoK, dat_dict["TempKEnv_parcel"] - degCtoK,
                where=cond1, alpha=0.3,  ##!!!
                color="r", hatch='XXX', edgecolor='k', zorder=zo)
            # Hatch areas of NEGATIVE Bouyancy
            if totalcape is True:
                cond2 = (dat_dict["tparcel"] < dat_dict["TempKEnv_parcel"]) & (dat_dict["pparcel"] <= dat_dict["P_el"])
            else:
                cond2 = (dat_dict["tparcel"] < dat_dict["TempKEnv_parcel"]) & (dat_dict["pparcel"] > dat_dict["plfc"])
            self.skewxaxis.fill_betweenx(
                dat_dict["pparcel"] / 100, dat_dict["tparcel"] - degCtoK, dat_dict["TempKEnv_parcel"] - degCtoK,
                where=cond2, alpha=0.3,  ###!!!! alpha
                color="b", hatch='///', edgecolor='r', zorder=zo)  ###color :none
        self.skewxaxis.legend(loc=2, prop={'family': 'SimHei', 'weight': 'normal', 'size': 12, })


def sys_call_arg(filename=configs.cfgs['SkewT_params']['path_input'],
             parceltype=configs.cfgs['SkewT_params']['parcel_type']):
    """实现外部调用作图"""
    file_save = os.path.splitext(os.path.basename(filename))[0] +"_"+ parceltype +".png"
    file_full_save = os.path.join(configs.cfgs['SkewT_params']['save_dir'], file_save)
    dat = ReadSoundingData(filename)
    dat_dict = plot_parcel_dat(filename, parceltype)
    plot = Sounding(dat)
    plot.make_skewt_axes()
    plot.add_profile()
    plot.lift_parcel(dat_dict)
    plt.savefig(file_full_save, ppi=300)

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("warning using!!! example: SkewTPlot filename 'ml', run configs path!")
        sys_call_arg()
    elif not os.path.exists(sys.argv[1]):
        print("file is not exist!!!")
    else:
        sys_call_arg(sys.argv[1])
        print("sucessful!!!")
