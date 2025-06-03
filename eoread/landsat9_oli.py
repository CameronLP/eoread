#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Landsat-9 OLI reader

Example:
    l1 = Level1_L9_OLI('LC09_L1TP_014034_20220618_20230411_02_T1/')

Data access:
    * https://earthexplorer.usgs.gov/
    * https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1
'''

import os
from glob import glob
import datetime
import tempfile
import numpy as np
import xarray as xr
import pyproj

from pathlib import Path
from datetime import datetime

from core import log, env
from core.tools import merge
from core.table import read_xml
from core.geo import n

from .common import DataArray_from_array


# Central wavelengths aren't described in metadata. Thus, they are hard-coded
wvl = {
    1:  442.96,
    2:  482.04,
    3:  561.41,
    4:  654.59,
    5:  864.67,
    6:  1608.86,
    7:  2200.73,
    8:  581,
    9:  1373.43,
    10: 10895,
    11: 12050,
}


def Level1_L9_OLI(dirname,
                  l9_angles=None,
                  chunks=500
                  ):
    '''
    Landsat-9 OLI reader.

    Arguments:
        dirname: name of the directory containing the Landsat9/OLI product
                 (Example: 'LC09_L1TP_014034_20220618_20230411_02_T1/')
        l9_angles: executable name of l9_angles program (ex: 'l9_angles/l9_angles'), used to generate the angles
                files automatically when missing, with the following command:
            l9_angles LC08_..._ANG.txt BOTH 1 -b 1
            l9_angles is available at:
            https://www.usgs.gov/land-resources/nli/landsat/solar-illumination-and-sensor-viewing-angle-coefficient-files

            It can be compiled with the following commands:
                wget https://landsat.usgs.gov/sites/default/files/documents/L9_ANGLES_2_7_0.tgz
                tar xzf L9_ANGLES_2_7_0.tgz
                rm -fv L9_ANGLES_2_7_0.tgz
                cd l9_angles
                make
                cd ..

    Returns a xr.Dataset
    '''
    ds = xr.Dataset()
    dirname = Path(dirname)
    assert dirname.exists()

    # Read metadata
    read_metadata(ds, dirname)
    if isinstance(chunks, int): chunks = [chunks]*2
    
    # define bands
    ds = ds.assign({n.bnames.name: ((n.bands.name), [f'B{b}' for b in wvl.keys()]),
                    n.cwav.name:((n.bands.name), list(wvl.values()))})

    # get datetime
    d = ds.attrs['IMAGE_ATTRIBUTES']['DATE_ACQUIRED']
    t = ds.attrs['IMAGE_ATTRIBUTES']['SCENE_CENTER_TIME']
    ds.attrs[n.datetime.name] = datetime.fromisoformat(d+'T'+t)

    # read_coordinates(ds, dirname, chunks)
    read_geometry(ds, dirname, l9_angles, chunks)
    ds = read_radiometry(ds, dirname, chunks)
    read_masks(ds, dirname, chunks)

    # other attributes
    proj = ds.attrs['PROJECTION_ATTRIBUTES']
    ds.attrs[n.crs.name]     = proj['ELLIPSOID'] + ' ' + str(proj['UTM_ZONE'])
    ds.attrs[n.platform]     = ds.attrs['IMAGE_ATTRIBUTES']['SPACECRAFT_ID']
    ds.attrs[n.sensor]       = ds.attrs['IMAGE_ATTRIBUTES']['SENSOR_ID']
    ds.attrs[n.product_name] = ds.attrs['PRODUCT_CONTENTS']['LANDSAT_PRODUCT_ID']
    
    ds = ds.rename({'y': n.rows.name, 'x': n.columns.name})   
    return ds.unify_chunks()


def read_metadata(ds, dirname):
    files_mtl = list(dirname.glob('LC*_MTL.xml'))
    assert len(files_mtl) == 1
    file_mtl = files_mtl[0]
    data_mtl = read_xml(file_mtl)
    ds.attrs.update(data_mtl)


def read_coordinates(ds, dirname, chunks):
    '''
    read lat/lon
    '''
    ds[n.lat.name] = DataArray_from_array(
        LATLON(dirname, 'lat'),
        ('y','x'),
        chunks,
    )
    ds[n.lon.name] = DataArray_from_array(
        LATLON(dirname, 'lon'),
        ('y','x'),
        chunks,
    )
    ds.attrs['totalheight'] = ds.y.size
    ds.attrs['totalwidth'] = ds.x.size


def gen_l9_angles(dirname, l9_angles=None):
    log.debug(f'Geometry file is missing in {dirname}, generating it with {l9_angles}...')
    angles_txt_file = list(dirname.glob('LC*_ANG.txt'))
    assert len(angles_txt_file) == 1
    assert l9_angles is not None
    assert os.path.exists(l9_angles)
    path_exe = os.path.abspath(l9_angles)
    path_angles = os.path.abspath(angles_txt_file[0])
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = f'cd {tmpdir} ; {path_exe} {path_angles} BOTH 1 -b 1'
        os.system(cmd)
        angle_files = os.path.join(tmpdir, '*')
        os.system(f'cp -v {angle_files} {dirname}')


def read_geometry(ds, dirname, l9_angles, chunks):
    
    # read sensor and solar angles
    for name, search in [(n.saa.name, 'LC*_SAA.TIF'),
                         (n.sza.name, 'LC*_SZA.TIF'),
                         (n.vaa.name, 'LC*_VAA.TIF'),
                         (n.vza.name, 'LC*_VZA.TIF')]:
        
        filenames = list(dirname.glob(search))
        if filenames != 1: continue
        data = xr.open_dataarray(filenames[0]).chunk([1]+chunks)
        ds[name] = (data.squeeze()/100).astype('float32')
    
    if (n.saa.name not in ds) and (l9_angles is not None):
        gen_l9_angles(dirname, l9_angles)


def read_radiometry(ds, dirname, chunks):
    
    rescale = ds.attrs['LEVEL1_RADIOMETRIC_RESCALING']
    thermal = ds.attrs['LEVEL1_THERMAL_CONSTANTS']
    
    # read radiances
    for b in ds[n.bnames.name].values:
        a = rescale[f'RADIANCE_ADD_BAND_{b[1:]}']
        m = rescale[f'RADIANCE_MULT_BAND_{b[1:]}']
        filenames = list(dirname.glob(f'LC*_{b}.TIF'))
        assert len(filenames) == 1
        data = xr.open_dataarray(filenames[0]).chunk([1]+chunks)
        ds[n.ltoa.name+f'_{b}'] = (m*data.squeeze()+a).astype('float32')
    
    ds = merge(ds, dim=n.bands.name, pattern=r'(.+)_B(.+)', dtype=str)
    
    # read reflectances
    for b in ds[n.bnames.name][:9].values:
        a = rescale[f'REFLECTANCE_ADD_BAND_{b[1:]}']
        m = rescale[f'REFLECTANCE_MULT_BAND_{b[1:]}']
        filenames = list(dirname.glob(f'LC*_{b}.TIF'))
        assert len(filenames) == 1
        data = xr.open_dataarray(filenames[0]).chunk([1]+chunks)
        ds[n.rtoa.name+f'_{b}'] = (m*data.squeeze()+a).astype('float32')
    
    ds = merge(ds, dim='bands_nvis', pattern=r'(.+)_B(.+)', dtype=str)
    
    
    # read brightness temperatures
    for b in ds[n.bnames.name][9:].values:
        k1 = thermal[f'K1_CONSTANT_BAND_{b[1:]}']
        k2 = thermal[f'K2_CONSTANT_BAND_{b[1:]}']
        rad = ds[n.ltoa.name].sel({n.bands.name:b[1:]})
        ds[n.bt.name+f'_{b}'] = (k2/np.log(k1/rad + 1)).astype('float32')

    ds = merge(ds, dim=n.bands_ir.name, pattern=r'(.+)_B(.+)', dtype=str)

    return ds

def read_masks(ds, dirname, chunks):
    for t in ['PIXEL','RADSAT']:
        filenames = list(dirname.glob(f'LC*_QA_{t}.TIF'))
        assert len(filenames) == 1
        ds[f'QA_{t}'] = xr.open_dataarray(filenames[0]).chunk([1]+chunks)

class LATLON:
    
    # FIXME: TO REVIZE  
    def __init__(self, dirname, kind, dtype='float32'):
        
        self.kind = kind

        files_B1 = glob(os.path.join(dirname, 'LC*_B1.TIF'))
        if len(files_B1) != 1:
            raise Exception('Invalid directory content ({})'.format(files_B1))
        file_B1 = files_B1[0]

        data = xr.open_dataarray(file_B1)

        height = data.y
        width = data.x
        self.shape = (height, width)

        gt = data.transform
        X0, X1 = (0, width-1)
        Y0, Y1 = (0, height-1)
        assert gt[1] == 0
        assert gt[3] == 0
        Xmin = gt[2] + X0*gt[0]
        Xmax = gt[2] + X1*gt[0]
        Ymin = gt[5] + Y0*gt[4]
        Ymax = gt[5] + Y1*gt[4]

        self.X = np.linspace(Xmin, Xmax, width)
        self.Y = np.linspace(Ymin, Ymax, height)
        self.dtype = np.dtype(dtype)
        
        self.latlon = pyproj.Proj("EPSG:4326") # WGS84
        self.utm = pyproj.Proj(data.crs)

        # see https://pyproj4.github.io/pyproj/stable/gotchas.html#upgrading-to-pyproj-2-from-pyproj-1
        self.transformer = pyproj.Transformer.from_proj(self.utm, self.latlon)

    def __getitem__(self, keys):
        x = self.X[keys[1]]
        y = self.Y[keys[0]]
        sx = (len(x),) if hasattr(x, '__len__') else ()
        sy = (len(y),) if hasattr(y, '__len__') else ()

        X, Y = np.meshgrid(x, y)
        lat, lon = self.transformer.transform(X, Y)

        if self.kind == 'lat':
            return lat.astype(self.dtype).reshape(sy+sx)
        else:
            return lon.astype(self.dtype).reshape(sy+sx)

def get_sample():
    return NotImplemented