from core import env, log
from core.tools import merge
from core.geo import n
from pathlib import Path

import numpy as np
import xarray as xr 
import dask.array as da


user_guide = 'https://ecostress.jpl.nasa.gov/downloads/userguides/1_ECOSTRESS_L1_UserGuide_20190619.pdf'

def Level1_ECOSTRESS(filepath: Path | str, chunks: int = 500):
    
    # Revize variables
    filepath = Path(filepath)
    log.debug('Reading h5file')
    data = xr.open_datatree(filepath, phony_dims='sort')
    raw = data['HDFEOS/GRIDS/ECO_L1CG_RAD_70m/Data Fields']
    raw = raw.to_dataset().chunk(chunks=chunks)
    
    # Read Metadata
    granule_mtd = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/ProductMetadata']
    attributes = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata']
    
    log.debug('parsing metadata text')
    info = data['HDFEOS INFORMATION']['StructMetadata.0'].values.item().decode()
    p = parser(info.split('\n'))
    p.parse()
    
    # Change radiometry of input data 
    log.debug('compute brightness temperature')
    l1 = transform_radiometry(raw, granule_mtd)   
    
    # Add attributes
    log.debug('add important attributes')
    for att in list(attributes):
        l1.attrs[att] = attributes[att].values.item()
    l1.attrs['hdfeos_info'] = p.data
    l1.attrs['user guide'] = user_guide
    
    # Change dimensions name and update coordinates
    new_dims = (n.rows.name,n.columns.name,n.bands_ir.name)    
    revize_dims = dict(zip(list(l1.dims), new_dims))
    l1 = l1.rename_dims(revize_dims)
    l1 = l1.assign({n.wav_ir.name: ((n.bands_ir.name),granule_mtd.BandSpecification.values[1:])})
    
    # Add latlon variables
    log.debug('add latlon variables')
    l1 = supplement_latlon(l1, chunks)
    return l1


def Level2_ECOSTRESS(filepath: Path | str, chunks: int = 500):
    
    # Revize variables
    filepath = Path(filepath)
    data = xr.open_datatree(filepath, phony_dims='sort')
    raw = data['HDFEOS/GRIDS/ECO_L2G_LSTE_70m/Data Fields']
    l1 = raw.to_dataset().chunk(chunks=chunks)
    
    # Read Metadata
    granule_mtd = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/ProductMetadata']
    attributes  = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata']
    
    info = data['HDFEOS INFORMATION']['StructMetadata.0'].values.item().decode()
    p = parser(info.split('\n'))
    p.parse()
    
    # Change radiometry of input data 
    l2 = transform_radiometry(raw, granule_mtd)   
    
    # Add attributes
    for att in list(attributes):
        l2.attrs[att] = attributes[att].values.item()
    l2.attrs['hdfeos_info'] = p.data
    l2.attrs['user guide'] = user_guide
    
    # Change dimensions name and update coordinates
    new_dims = [n.rows.name,n.columns.name,n.bands_ir.name]
    coords = {n.bands_ir.name: granule_mtd.BandSpecification.values[1:]}
    
    revize_dims = dict(zip(list(l2.dims), new_dims))
    l2 = l2.rename_dims(revize_dims)
    l2 = l2.assign_coords(coords)
    
    # Add latlon variables
    l2 = supplement_latlon(l2, chunks)
    return l2


def transform_radiometry(raw_data, granule_mtd):
    
    # Combine band radiances into a single variable 
    level1 = merge(raw_data, dim=n.bands.name, pattern=r'(.+)_(\d+)')
    
    # Compute brightness temperature for Emissive bands 
    level1[n.bt.name] = compute_bt(level1, granule_mtd)
    level1[n.bt.name].attrs = {}
    level1[n.bt.name].attrs['units'] = 'Kelvin'
        
    level1 = level1.drop_indexes(list(level1.coords)) \
                   .reset_coords(drop=True)
    return level1

def supplement_latlon(l1, chunks): 
        
    # Compute LatLon variables
    size = l1['cloud'].shape
    north, south = l1.NorthBoundingCoordinate, l1.SouthBoundingCoordinate
    east, west = l1.EastBoundingCoordinate, l1.WestBoundingCoordinate
    
    dims = [n.rows.name,n.columns.name]
    lat = da.linspace(south,north,size[1]).reshape((1,size[1]))
    lon = da.linspace(west,east,size[0]).reshape((size[0],1))
    l1[n.lon.name] = xr.DataArray(da.repeat(lon, size[1], axis=1), 
                                  dims=dims).chunk(chunks=chunks)
    l1[n.lat.name] = xr.DataArray(da.repeat(lat, size[0], axis=0), 
                                  dims=dims).chunk(chunks=chunks)
    return l1

def compute_bt(l1, granule_mtd) -> xr.DataArray:
    """Calibration for the emissive channels."""
    # Initialized constants
    K1 = 1.191042 * 1e8
    K2 = 1.4387752 * 1e4
    
    # Temperature correction
    cwvl   = granule_mtd.BandSpecification[1:].rename(phony_dim_0='bands') * 1e-3 # convert into µm
    gain   = granule_mtd.CalibrationGainCorrection.rename(phony_dim_1='bands')
    offset = granule_mtd.CalibrationOffsetCorrection.rename(phony_dim_1='bands')
    l1 = l1.assign({n.cwav.name: ((n.bands_ir.name),cwvl.data)})
    
    # Some versions of the modis files do not contain all the bands.
    valid = ~l1['radiance'].isnull()
    array = K2 / (cwvl * np.log(K1 / (l1['radiance'].where(valid) * cwvl ** 5) + 1))
    return gain * array.where(valid) + offset
    

def get_sample(level:int=1, use_cache:bool=True) -> Path:
    try: 
        from core.files import cache_dataframe
        from sand.nasa import DownloadNASA
        from sand.sample_product import products
    except ImportError:
        log.error('To use get_sample function, you need to install SAND module',
                  e=ImportError)
        
    cachefile = env.getdir('DIR_STATIC')/'query_ecostress.pickle'
    if use_cache: cache_deco = cache_dataframe(cachefile)
    else: cache_deco = lambda x: x
    
    sensor = 'ECOSTRESS'
    params = products[sensor][f'level{level}']
    dl = DownloadNASA(sensor, level)
    ls = cache_deco(dl.query)(**params)
    return dl.download(ls.iloc[0], env.getdir('DIR_SAMPLES'))

class parser:
    
    def __init__(self, text: list):
        self.data = {}
        self.text = text.copy()
    
    def empty(self): return len(self.text) == 0
    
    def consume(self):
        
        line = self.text[0]
        if len(self.text) == 1: # case for last line
            self.text = []
            return line.strip()
                
        self.text = self.text[1:]
        
        line = line.strip()
        while line == "":
            line = self.consume()
        
        return line.strip()
    
    def peek(self):
        return self.text[0].strip()
    
    def parse(self):
        
        while not self.empty():
            end = self._parse_recu(self.data)
            if end: break
            
    def _parse_recu(self, data: dict=None): 
        
        line = self.consume()
        if line == "END":
            return True
        
        key, val = line.split('=')
        if key in ['GROUP','OBJECT']: 
            data[val] = {}
            
            line = self.peek()
            while f'END_' not in line:
                self._parse_recu(data[val])
                line = self.peek() # refresh peeked line !! 
            
            # closing tag
            self.consume()
        
        else: data[key] = val 
        
        return False