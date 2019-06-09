import sys
from cx_Freeze import setup, Executable
import os
import scipy
import matplotlib
from mpl_toolkits import basemap

includefiles_list=["C:\\Users\zy\\Anaconda3\\DLLs\\tcl86t.dll", "C:\\Users\\zy\\Anaconda3\\DLLs\\tk86t.dll",
'C:\\Users\\zy\\Anaconda3\\Lib\\site-packages\\mpl_toolkits',
"C:\\Users\\zy\\Anaconda3\\Lib\\site-packages\\pyproj","C:\\Users\\zy\\Anaconda3\\Lib\site-packages\_geoslib.cp36-win_amd64.pyd"]
##                   r'C:\Windows\WinSxS\Manifests\x86_microsoft.vc90.crt_1fc8b3b9a1e18e3b_9.0.30729.9279_none_50939ec6bcb7c97c.manifest',
##                   r'?C:\Windows\WinSxS\x86_microsoft.vc90.crt_1fc8b3b9a1e18e3b_9.0.30729.9279_none_50939ec6bcb7c97c\msvcm90.dll',
##                   r'?C:\Windows\WinSxS\x86_microsoft.vc90.crt_1fc8b3b9a1e18e3b_9.0.30729.9279_none_50939ec6bcb7c97c\msvcp90.dll',
##                   r'?C:\Windows\WinSxS\x86_microsoft.vc90.crt_1fc8b3b9a1e18e3b_9.0.30729.9279_none_50939ec6bcb7c97c\msvcr90.dll']
scipy_path = os.path.dirname(scipy.__file__)
includefiles_list.append(scipy_path)
includefiles_list.append((basemap.basemap_datadir,"data"))

os.environ['TCL_LIBRARY'] = r'C:\Users\zy\Anaconda3\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\zy\Anaconda3\tcl\tk8.6'
addtional_mods = ['numpy.core._methods', 'numpy.lib.format']

buildOptions = dict(packages = ['matplotlib.backends.backend_tkagg',\
                                'tkinter','numpy.matlib','_tkinter','osgeo._gdal','cytoolz'],\
                    excludes = [],includes = addtional_mods,
                    include_files = includefiles_list,)
options = {
'build_exe': {'path': sys.path + ['modules']}
}
   
base = None
if sys.platform == "win32":
    base = "Win32GUI"

executables = [
	Executable('Advection_parameter.py', base = base),
	Executable('Advection_Plot.py', base = base),
	Executable('SkewT_parameter.py', base = base),
    Executable('SkewT_Plot.py', base = base),]
    # Executable('./libs/VelocityDiagnosis.py', base=base),
    # Executable('./libs/dynamic.py', base=base),
    # Executable('./libs/diagnosis.py', base=base),
	# Executable('./configs/__init__.py', base=base),]

setup(name='T-lnP',
      version = '0.1',
      description = 'atmospheric physic processing',
      author = 'zhengyu',
      options = dict(build_exe = buildOptions),
      executables = executables)
