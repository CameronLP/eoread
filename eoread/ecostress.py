from eoread.utils import filter_metadata
from core.tools import merge
from core import env, log
from core.geo import n
from pathlib import Path
from shapely import wkt
from typing import Union

import numpy as np
import xarray as xr 
import dask.array as da


user_guide = 'https://ecostress.jpl.nasa.gov/downloads/userguides/1_ECOSTRESS_L1_UserGuide_20190619.pdf'

def Level1_ECOSTRESS(
        filepath: Union[Path, str], 
        chunks: Union[int, list] = 500, 
        metadata_template: Union[list, None] = None,
        v1_compat: bool = False
    ) -> xr.Dataset:
    """
    Read an ECOSTRESS Level1 product as an xarray.Dataset.
    
    ECOSTRESS (Ecosystem Spaceborne Thermal Radiometer Experiment on Space Station)
    provides high-resolution thermal infrared measurements across 5 spectral bands.

    Args:
        filepath: Path to the ECOSTRESS HDF5 file (.h5)
        chunks: Size of chunks for spatial dimensions. If int, applies to both dimensions.
                If list, should be [rows_chunk, columns_chunk]
        metadata_template: List of metadata keys to include. If None, includes all metadata.
                          Use empty list [] for minimal metadata.
        v1_compat: If True, formats output to match version 1 structure for backward compatibility
        
    Example:
        >>> ds = Level1_ECOSTRESS('ECOv002_L1CG_RAD_*.h5', chunks=1000)
    """
    
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
    l1 = l1.assign_coords({
        str(n.bands): l1[str(n.bands)].values.astype(str),
        str(n.bgroup): (str(n.bands), ['bands_ir']*5)
    })
    
    # Add latlon variables
    log.debug('add latlon variables')
    l1 = _supplement_latlon(l1, list(chunks))
    
    if v1_compat: return _v1_compat(l1)
    else: return l1


def Level2_ECOSTRESS(
        filepath: Union[Path, str], 
        chunks: int = 500
    ) -> xr.Dataset:
    """
    Read an ECOSTRESS Level2 product as an xarray.Dataset.
    
    Processes Level2 Land Surface Temperature and Emissivity (LSTE) data.

    Args:
        filepath: Path to the ECOSTRESS Level2 HDF5 file (.h5)
        chunks: Size of chunks for spatial dimensions

    Returns:
        xarray.Dataset containing:
            - Surface temperature and emissivity products
            - Quality masks and flags
            - Geolocation arrays (lat, lon)
            - Metadata attributes
    """
    # Revise variables
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
    new_dims = [str(n.rows),str(n.columns)]    
    revize_dims = dict(zip(list(l2.dims), new_dims))
    l2 = l2.rename_dims(revize_dims)
    
    # Add latlon variables
    l2 = _supplement_latlon(l2, chunks)
    return l2


def _transform_radiometry(raw_data: xr.Dataset, granule_mtd: xr.Dataset) -> xr.Dataset:
    """Convert raw radiances to calibrated units and compute brightness temperature."""
    # Combine band radiances into a single variable 
    level1 = merge(raw_data, dim=str(n.bands), pattern=r'(.+)_(\d+)')
    
    # Rename radiance variable
    level1 = level1.rename({'radiance': str(n.ltoa)})
    level1[str(n.ltoa)].attrs['unit'] = 'W/sr/m^2'
    
    # Compute brightness temperature for Emissive bands 
    return _compute_bt(level1, granule_mtd)

def _supplement_latlon(l1: xr.Dataset, chunks: list) -> xr.Dataset:
    """Add latitude and longitude coordinates based on scene boundary."""
    # Compute LatLon variables
    size = l1['cloud'].shape
    poly = wkt.loads(l1.metadata['SceneBoundaryLatLonWKT'])
    coords = np.array(poly.exterior.coords)
    north  = coords[:,1].max()
    south  = coords[:,1].min()
    east   = coords[:,0].max()
    west   = coords[:,0].min()
    
    # Build the lat and lon arrays
    dims = [str(n.rows),str(n.columns)]
    lat = da.linspace(north,south,size[0]).reshape((size[0],1))
    lon = da.linspace(west,east,size[1]).reshape((1,size[1]))
    l1[str(n.lon)] = xr.DataArray(da.repeat(lon, size[0], axis=0), 
                                  dims=dims).chunk(chunks=chunks)
    l1[str(n.lat)] = xr.DataArray(da.repeat(lat, size[1], axis=1), 
                                  dims=dims).chunk(chunks=chunks)
    return l1

def _compute_bt(l1: xr.Dataset, granule_mtd: xr.Dataset) -> xr.Dataset:
    """Compute brightness temperature from radiance using Planck's law."""
    # Initialized constants
    K1 = 1.191042 * 1e8
    K2 = 1.4387752 * 1e4
    
    # Temperature correction
    cwvl   = granule_mtd.BandSpecification[1:].rename(phony_dim_0=str(n.bands)) # convert into µm
    gain   = granule_mtd.CalibrationGainCorrection.rename(phony_dim_1=str(n.bands))
    offset = granule_mtd.CalibrationOffsetCorrection.rename(phony_dim_1=str(n.bands))
    l1 = l1.assign({str(n.cwav): ((str(n.bands)), cwvl.data*1e3)})
    
    # Some versions of the modis files do not contain all the bands.
    valid = ~l1[str(n.ltoa)].isnull()
    array = K2 / (cwvl * np.log(K1 / (l1[str(n.ltoa)].where(valid) * cwvl ** 5) + 1))
    l1[str(n.bt)] = gain * array.where(valid) + offset
    l1[str(n.bt)].attrs = {'unit': 'Kelvin'}
    
    return l1

def _v1_compat(ds: xr.Dataset) -> xr.Dataset:
    """Transform dataset to version 1 format for backward compatibility."""
    
    # Assign central wavelengths as bands coordinates
    ds = ds.assign_coords({str(n.bands): [8290,8780,9200,10490,12090]})
    ds = ds.rename({str(n.bands): 'bands_tir'})
    
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


def get_sample(level: int = 1) -> Path:
    """
    Download or retrieve a sample ECOSTRESS product for testing.
    
    Requires the 'sand' module for NASA data access.

    Args:
        level: Processing level of the product (1 or 2). Level 1 provides
               radiance/brightness temperature, Level 2 provides surface temperature.

    Returns:
        Path to the downloaded ECOSTRESS HDF5 file
        
    Raises:
        ImportError: If the 'sand' module is not installed
    """
    try: 
        from sand.nasa import DownloadNASA
        from sand.sample_product import products
    except ImportError:
        raise ImportError('To use get_sample function, you need to install SAND module')
    
    sensor = 'ISS-ECOSTRESS'
    prod_id = products[sensor][f'l{level}_product']
    target = env.getdir('DIR_SAMPLES')/prod_id
    dl = DownloadNASA()
    dl.download_file(target.name, target.parent)
    assert target.exists()
    return target

class _parser:
    """Parser for HDFEOS metadata structure."""
    
    def __init__(self, text: list):
        """Initialize parser with metadata text lines."""
        self.data = {}
        self.text = text.copy()
    
    def empty(self) -> bool:
        """Check if there are remaining lines to parse."""
        return len(self.text) == 0
    
    def consume(self) -> str:
        """Remove and return the next non-empty line."""
        
        line = self.text[0]
        if len(self.text) == 1: # case for last line
            self.text = []
            return line.strip()
                
        self.text = self.text[1:]
        
        line = line.strip()
        while line == "":
            line = self.consume()
        
        return line.strip()
    
    def peek(self) -> str:
        """Return the next line without consuming it."""
        return self.text[0].strip()
    
    def parse(self) -> None:
        """Parse all remaining lines into structured metadata."""
        
        while not self.empty():
            end = self._parse_recu(self.data)
            if end: break
            
    def _parse_recu(self, data: dict = None) -> bool:
        """Recursively parse metadata groups and objects."""
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