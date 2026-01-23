from eoread.utils import filter_metadata
from core.tools import merge
from core import env, log
from core.geo import n
from pathlib import Path
from shapely import wkt

import numpy as np
import xarray as xr 
import dask.array as da


user_guide = 'https://ecostress.jpl.nasa.gov/downloads/userguides/1_ECOSTRESS_L1_UserGuide_20190619.pdf'

def Level1_ECOSTRESS(
        filepath: Path | str, 
        chunks: int|list = 500, 
        metadata_template: list|None = None,
        v1_compat: bool = False
    ):
    '''
    Read an ECOSTRESS Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, brightness temperatures,
    the angles on the full grid, etc.

    Arguments:
        filepath: Path of the ECOSTRESS H5file
        chunks: Size of chunks for spatial axis
        metadata_template: If None, add all metadata in output xarray.Dataset attributes else add only specified metadata.
        v1_compat: Option to format output xarray.Dataset such as version 1
    '''
    
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'
    if isinstance(chunks, int):
        chunks = [chunks]*2    
    
    # Revize variables    
    log.debug('Reading h5file')
    data = xr.open_datatree(filepath, phony_dims='sort', engine='h5netcdf')
    raw = data['HDFEOS/GRIDS/ECO_L1CG_RAD_70m/Data Fields']
    raw = raw.to_dataset().chunk(chunks=dict(zip(list(raw.dims), chunks)))
    
    # Read Metadata
    granule_mtd = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/ProductMetadata']
    attributes = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata']
    
    log.debug('parsing metadata text')
    info = data['HDFEOS INFORMATION']['StructMetadata.0'].values.item().decode()
    p = _parser(info.split('\n'))
    p.parse()
    
    # Change radiometry of input data 
    log.debug('compute brightness temperature')
    l1 = _transform_radiometry(raw, granule_mtd)   
    
    # Add attributes
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    log.debug('add important attributes')
    l1.attrs['metadata'] = {k: v.item() for k,v in attributes.items()}
    l1.attrs['metadata'] = filter_fn(l1.attrs['metadata'], metadata_template)
    l1.attrs['hdfeos_info'] = filter_fn(p.data, metadata_template)
    l1.attrs['user_guide'] = user_guide
    
    # Add general information
    l1.attrs[str(n.datetime)] = l1.metadata['ProductionDateTime']
    l1.attrs[str(n.platform)] = l1.metadata['PlatformShortName']
    l1.attrs[str(n.sensor)] = l1.metadata['InstrumentShortName']
    l1.attrs[str(n.product_name)] = filepath.name
    l1.attrs[str(n.input_directory)] = str(filepath.parent)
    l1.attrs[str(n.resolution)] = 70
    
    # Change dimensions name and update coordinates
    new_dims = (str(n.rows),str(n.columns))    
    revize_dims = dict(zip(list(l1.dims)[:-1], new_dims))
    l1 = l1.rename_dims(revize_dims)
    l1 = l1.assign({
        str(n.cwav): ((str(n.bands)), granule_mtd.BandSpecification.values[1:]*1e3),
        str(n.bnames): ((str(n.bands)), l1[str(n.bands)].values.astype(str))
    })
    
    # Add latlon variables
    log.debug('add latlon variables')
    l1 = _supplement_latlon(l1, list(chunks))
    
    if v1_compat: return _v1_compat(l1)
    else: return l1


def Level2_ECOSTRESS(filepath: Path | str, chunks: int = 500):
    
    # Revize variables
    filepath = Path(filepath)
    data = xr.open_datatree(filepath, phony_dims='sort', engine='h5netcdf')
    raw = data['HDFEOS/GRIDS/ECO_L2G_LSTE_70m/Data Fields']
    l2 = raw.to_dataset().chunk(chunks=chunks)
    
    # Read Metadata
    granule_mtd = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/ProductMetadata']
    attributes  = data['HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata']
    
    info = data['HDFEOS INFORMATION']['StructMetadata.0'].values.item().decode()
    p = _parser(info.split('\n'))
    p.parse()
    
    # Change radiometry of input data 
    l2 = _transform_radiometry(raw, granule_mtd)   
    
    # Add attributes
    for att in list(attributes):
        l2.attrs[att] = attributes[att].values.item()
    l2.attrs['hdfeos_info'] = p.data
    l2.attrs['user guide'] = user_guide
    
    # Change dimensions name and update coordinates
    new_dims = [str(n.rows),str(n.columns),str(n.bands_ir)]
    coords = {str(n.bands_ir): granule_mtd.BandSpecification.values[1:]}
    
    revize_dims = dict(zip(list(l2.dims), new_dims))
    l2 = l2.rename_dims(revize_dims)
    l2 = l2.assign_coords(coords)
    
    # Add latlon variables
    l2 = _supplement_latlon(l2, chunks)
    return l2


def _transform_radiometry(raw_data, granule_mtd):
    
    # Combine band radiances into a single variable 
    level1 = merge(raw_data, dim=str(n.bands), pattern=r'(.+)_(\d+)')
    
    # Rename radiance variable
    level1 = level1.rename({'radiance': str(n.ltoa)})
    level1[str(n.ltoa)].attrs['unit'] = 'W/sr/m^2'
    
    # Compute brightness temperature for Emissive bands 
    level1[str(n.bt)] = _compute_bt(level1, granule_mtd)
    level1[str(n.bt)].attrs = {}
    level1[str(n.bt)].attrs['unit'] = 'Kelvin'
    
    return level1

def _supplement_latlon(l1, chunks): 
        
    # Compute LatLon variables
    size = l1['cloud'].shape
    poly = wkt.loads(l1.metadata['SceneBoundaryLatLonWKT'])
    coords = np.array(poly.exterior.coords)
    north  = coords[:,1].max()
    south  = coords[:,1].min()
    east   = coords[:,0].max()
    west   = coords[:,0].min()
    
    dims = [str(n.rows),str(n.columns)]
    lat = da.linspace(north,south,size[0]).reshape((size[0],1))
    lon = da.linspace(west,east,size[1]).reshape((1,size[1]))
    l1[str(n.lon)] = xr.DataArray(da.repeat(lon, size[0], axis=0), 
                                  dims=dims).chunk(chunks=chunks)
    l1[str(n.lat)] = xr.DataArray(da.repeat(lat, size[1], axis=1), 
                                  dims=dims).chunk(chunks=chunks)
    return l1

def _compute_bt(l1, granule_mtd) -> xr.DataArray:
    """Calibration for the emissive channels."""
    # Initialized constants
    K1 = 1.191042 * 1e8
    K2 = 1.4387752 * 1e4
    
    # Temperature correction
    cwvl   = granule_mtd.BandSpecification[1:].rename(phony_dim_0=str(n.bands)) # convert into µm
    gain   = granule_mtd.CalibrationGainCorrection.rename(phony_dim_1=str(n.bands))
    offset = granule_mtd.CalibrationOffsetCorrection.rename(phony_dim_1=str(n.bands))
    l1 = l1.assign({str(n.cwav): ((str(n.bands_ir)),cwvl.data)})
    
    # Some versions of the modis files do not contain all the bands.
    valid = ~l1[str(n.ltoa)].isnull()
    array = K2 / (cwvl * np.log(K1 / (l1[str(n.ltoa)].where(valid) * cwvl ** 5) + 1))
    bt = gain * array.where(valid) + offset
    return bt.rename({str(n.bands): str(n.bands_ir)})

def _v1_compat(ds):
    """Ensure back-compatibility"""
    
    # Rename IR bands into bands_tir
    ds = ds.rename({str(n.bands_ir): 'bands_tir'})
    
    # Assign central wavelengths as bands coordinates
    ds = ds.assign_coords({'bands_tir': [8290,8780,9200,10490,12090],
                           str(n.bands): [8290,8780,9200,10490,12090]})
    
    # Add a new variables called flags
    flags = ds.data_quality[0] != 0
    ds['flags'] = flags.astype(n.flags.dtype)
    
    # Add missing attributes
    ds.attrs['product_name'] = ds.attrs['product_name'][:-3]
    ds.attrs['Description'] = ds.metadata['LongName']
    ds.attrs['shortname']  = str(ds.metadata['ShortName'])
    ds.attrs['night']      = str(str(ds.metadata['DayNightFlag']) != 'Day')
    ds.attrs['CRS']        = str(ds.metadata['CRS'])
    ds.attrs['Boundary']   = str(ds.metadata['SceneBoundaryLatLonWKT'])
    ds.attrs['version']    = str(ds.metadata['PGEVersion'])  

    return ds


def get_sample(level: int=1) -> Path:
    """
    Bring a ECOSTRESS file path to test reading function

    Args:
        level (int, optional): Level of the product. Defaults to 1.
    """
    try: 
        from sand.nasa import DownloadNASA
        from sand.sample_product import products
    except ImportError:
        raise ImportError('To use get_sample function, you need to install SAND module')
    
    sensor = 'ISS-ECOSTRESS'
    prod_id = products[sensor][f'l{level}_product']
    target = env.getdir('DIR_SAMPLES')/(prod_id+'.h5')
    if not target.exists():
        # TODO: remove when SAND's download_file supports filegen
        dl = DownloadNASA()
        dl.download_file(prod_id, env.getdir('DIR_SAMPLES'))
    assert target.exists()
    return target

class _parser:
    
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