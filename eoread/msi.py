#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Update processing baseline 4.00
# https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-upcoming

from pathlib import Path
from typing import Literal, Union

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from core.tools import merge, drop_unused_dims
from core.table import read_xml
from core import env, log
from core.geo import n
from core.interpolate import interp, Linear

from eoread.utils import filter_metadata, spatial_resample
from eoread.common import DataArray_from_array


user_guide = 'https://sentinels.copernicus.eu/documents/247904/685211/Sentinel-2_User_Handbook.pdf/8869acdf-fd84-43ec-ae8c-3e80a436a16c?t=1438278087000'

def Level1_MSI(
        dirname: Union[str, Path],
        chunks: Union[int, tuple] = 500,
        resolution: Literal[10,20,60,None] = 10,
        metadata_template: Union[list, None] = None, 
        v1_compat: bool = False,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a Sentinel-2 MSI Level1C product as an xarray.Dataset.
    
    MSI (MultiSpectral Instrument) provides 13 spectral bands from visible to SWIR
    with spatial resolutions of 10m, 20m, and 60m.

    Args:
        dirname: Path to the Sentinel-2 .SAFE directory
        chunks: Size of chunks for spatial dimensions. If int, applies to both dimensions.
                If tuple, should be (rows_chunk, columns_chunk)
        resolution: Resample all bands to provided resolution and concatenate.
                If None, keep original resolutions (10m, 20m, 60m) separate.
        metadata_template: List of metadata keys to include. If None, includes all metadata.
        v1_compat: If True, formats output to match version 1 structure
    
    Example:
        >>> ds = Level1_MSI('S2A_MSIL1C_*.SAFE', chunks=1000)
    """
    ds = xr.Dataset()
    dirname = Path(dirname).resolve()
    if isinstance(chunks, int): chunks = [chunks]*2

    if list(dirname.glob('GRANULE')):
        granules = list((dirname/'GRANULE').glob('*'))
        assert len(granules) == 1
        granule_dir = granules[0]
    else: granule_dir = dirname

    # load xml files
    xmlgranule = granule_dir/'MTD_TL.xml'
    xmlroot = dirname/'MTD_MSIL1C.xml'
    assert xmlgranule.exists()
    assert xmlroot.exists()
    
    if verbose: log.debug('Reading metadata files')
    xmlgranule = read_xml(xmlgranule)
    xmlroot = read_xml(xmlroot)

    # load main xml file
    product_image = xmlroot['General_Info']['Product_Image_Characteristics']
    processing_baseline = xmlroot['General_Info']['Product_Info']['PROCESSING_BASELINE']
    
    # Extract bands wavelength
    metadata = dict(cwvl=[], resolution=[], name=[])
    if verbose: log.debug('Extract central wavelength')
    for spec in product_image['Spectral_Information_List']['Spectral_Information']:
        metadata['cwvl'].append(spec['Wavelength']['CENTRAL']['values'])
        metadata['name'].append(spec['attributes']['physicalBand'])
        metadata['resolution'].append(spec['RESOLUTION'])

    # get platform
    tile_id = xmlgranule['General_Info']['TILE_ID']['values']
    platform = tile_id[:3]
    assert platform in ['S2A', 'S2B', 'S2C']

    # read image size for current resolution
    res = str(resolution) if resolution else '10'
    geocoding = xmlgranule['Geometric_Info']['Tile_Geocoding']
    for e in geocoding.get('Size'):
        if e['attributes']['resolution'] == res:
            ds.attrs['totalheight'] = e.get('NROWS')
            ds.attrs['totalwidth'] = e.get('NCOLS')
            break

    # attributes
    if verbose: log.debug('Add important attributes')
    sensing_time = xmlgranule['General_Info']['SENSING_TIME']['values']
    ds.attrs[str(n.datetime)] = sensing_time
    ds.attrs[str(n.platform)] = {
        "S2A": "Sentinel-2A",
        "S2B": "Sentinel-2B",
        "S2C": "Sentinel-2C",
    }[platform]
    ds.attrs[str(n.resolution)] = resolution
    ds.attrs[str(n.sensor)] = 'MSI'
    ds.attrs[str(n.product_name)] = dirname.name
    ds.attrs[str(n.input_directory)] = str(dirname.parent)
    ds.attrs['user_guide'] = user_guide

    # lat-lon
    if verbose: log.debug('Extract central wavelength')
    _msi_read_latlon(ds, chunks, xmlgranule)

    # msi_read_geometry
    if verbose: log.debug('Read and compute geometric angles')
    tileangles = xmlgranule['Geometric_Info']['Tile_Angles']
    ds = _msi_read_geometry(ds, tileangles, chunks)
    
    # msi read quality mask
    if verbose: log.debug('WARNING: SKIPPING >> Read quality masks')
    # ds = _msi_read_qi(ds, granule_dir, chunks)

    # msi_read_toa and quality masks
    if verbose: log.debug('Read top of atmosphere data')
    ds = _msi_read_toa(ds, granule_dir, processing_baseline, product_image, chunks, metadata, resolution)
    ds = ds.assign({str(n.cwav): ((str(n.bands)), metadata['cwvl'])})
    ds = ds.assign_coords({str(n.bands): metadata['name']})
    
    # Filter metadata
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs['metadata_granule'] = filter_fn(xmlgranule, metadata_template)
    ds.attrs['metadata'] = filter_fn(xmlroot, metadata_template)

    ds = drop_unused_dims(ds)
    if resolution:
        ds = ds.set_coords(str(n.bgroup))
    
    # SRF getter
    ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_eumetsat'
    ds.attrs['_srf_getter_arg'] = {
        'S2A': 'sentinel2_1_msi', 
        'S2B': 'sentinel2_2_msi', 
        'S2C': 'sentinel2_3_msi', 
    }[platform]
    
    if v1_compat: return _v1_compat(ds)
    return ds.unify_chunks()
    

def _msi_read_latlon(ds: xr.Dataset, chunks: list, xmlgranule: dict) -> None:
    """Add latitude and longitude arrays from UTM projection."""
    dims = (str(n.rows), str(n.columns))
    geocoding = xmlgranule['Geometric_Info']['Tile_Geocoding']
    
    # FIXME: LATLON should be replaced by a map_block
    ds[str(n.lat)] = DataArray_from_array(
        _LATLON(geocoding, 'lat', ds), dims,
        chunks=chunks,
    )

    ds[str(n.lon)] = DataArray_from_array(
        _LATLON(geocoding, 'lon', ds), dims,
        chunks=chunks,
    )

def _msi_read_qi(ds: xr.Dataset, granule_dir: Path, chunks: list) -> xr.Dataset:
    """Read quality indicator masks from QI_DATA directory."""
    for filename in (granule_dir/'QI_DATA').glob(f'*.jp2'):
        
        if '_PVI' in filename.stem: continue
        arr = xr.open_dataarray(filename, engine='rasterio')
        arr = arr.chunk([1]+list(chunks))
        arr = arr.rename(x='x_red', y='y_red').astype('float32')
        ds[filename.stem] = arr.rename({'band': str(n.detector)})
    
    ds = ds.rename_vars({'MSK_CLASSI_B00':'MSK_CLASSI'})
    ds = merge(ds, dim=str(n.bands), pattern=r'(.+)_B(.+)', dtype=str)

    return ds

def _msi_read_toa(
        ds: xr.Dataset, 
        granule_dir: Path, 
        processing_baseline: str, 
        product_image: dict, 
        chunks: list, 
        metadata: dict, 
        resolution: int
    ) -> xr.Dataset:
    """Read and calibrate TOA reflectance from JP2 files.
    
    Applies radiometric offset correction (baseline >= 4.0) and quantification.
    Optionally resamples all bands to 10m resolution if concat=True.
    """
    # Retrieve radiometric offset
    if float(processing_baseline) >= 4:
        radio_offset = [
            int(x['values'])
            for x in product_image['Radiometric_Offset_List']['RADIO_ADD_OFFSET']]
    else: 
        radio_offset = [0]*len(metadata['name'])
    
    # Open deserved bands
    indexes = []
    quantif = product_image['QUANTIFICATION_VALUE']['values']
    for filename in (granule_dir/'IMG_DATA').glob(f'*.jp2'):
        
        # Add band to dataset
        band = filename.stem.split('_')[-1]
        if 'TCI' == band: continue
        iband = list(metadata['name']).index(band.replace('B0','B'))
        indexes.append(iband)
        
        arr = xr.open_dataarray(filename, engine='rasterio').chunk([1]+list(chunks))
        arr = ((arr+radio_offset[iband])/quantif).astype('float32')
        
        # Resample the array
        if resolution:
            ratio = {str(n.columns): ds.totalheight, str(n.rows): ds.totalwidth} 
            arr_resampled = spatial_resample(arr.squeeze(), ratio, chunks)
            ds[str(n.rtoa)+f'_{band}'] = arr_resampled
        else:
            res = str(metadata['resolution'][iband]) + 'm'
            name = str(n.rtoa)+f'_{res}_{band}'
            arr = arr.squeeze().rename({'y': f'y_{res}', 'x': f'x_{res}'})
            ds[name] = arr.chunk(chunks)
    
    if resolution:
        res = [n.bands+f'_{r}m' for r in metadata['resolution']]
        ds = merge(ds, dim=str(n.bands), pattern=r'(.+)_B(.+)', dtype=str)
        ds[str(n.rtoa)].attrs.update(unit=None)
        ds = ds.assign({str(n.bgroup): (str(n.bands), res)})
    else:
        ds = merge(ds, dim=str(n.bands+'_10m'), pattern=r'(.+_10m)_B(.+)', dtype=str)
        ds = merge(ds, dim=str(n.bands+'_20m'), pattern=r'(.+_20m)_B(.+)', dtype=str)
        ds = merge(ds, dim=str(n.bands+'_60m'), pattern=r'(.+_60m)_B(.+)', dtype=str)
        ds[str(n.rtoa)+'_10m'].attrs.update(unit=None)
        ds[str(n.rtoa)+'_20m'].attrs.update(unit=None)
        ds[str(n.rtoa)+'_60m'].attrs.update(unit=None)
    
    return ds


def _msi_read_geometry(
        ds: xr.Dataset, 
        tileangles: dict, 
        chunks: tuple
    ) -> xr.Dataset:
    """
    Read and interpolate geometric angles from tie points to full resolution.

    Args:
        ds: Input dataset with product dimensions
        tileangles: Dictionary containing XML blocks for solar and view angles
        chunks: Chunk sizes for dask arrays

    Returns:
        Dataset with added angle variables (sza, saa, vza, vaa) and their tie-point versions
    """
    
    # read solar angles at tiepoints
    dims = ('tie_rows', 'tie_columns')
    sza = _read_xml_block(tileangles['Sun_Angles_Grid']['Zenith'], dims)
    saa = _read_xml_block(tileangles['Sun_Angles_Grid']['Azimuth'], dims)

    shp = (ds.totalheight, ds.totalwidth)

    # read view angles (for each band)
    tie_shape = None
    vza, vaa = {}, {}
    for e in tileangles.get('Viewing_Incidence_Angles_Grids'):

        # Reading zenith angles
        data: np.ndarray = _read_xml_block(e['Zenith'], dims)
        bandid = int(e['attributes']['bandId'])
        
        if tie_shape is None: tie_shape = data.shape # in case the size is not constant
        data = data.values.flatten()
        
        if bandid not in vza: vza[bandid] = data
        valid = ~np.isnan(data) # indexes where the data is not null
        vza[bandid][valid] = data[valid]

        # Reading azimuth angles
        data = _read_xml_block(e['Azimuth'], dims)
        bandid = int(e['attributes']['bandId'])
        
        data: np.ndarray = data.values.flatten()
        
        if bandid not in vaa: vaa[bandid] = data
        valid = ~np.isnan(data) # indexes where the data is not null
        vaa[bandid][valid] = data[valid]

    # reshape to original 
    for b in vza:
        vza[b] = vza[b].reshape(tie_shape)
    for b in vaa:
        vaa[b] = vaa[b].reshape(tie_shape)
    

    # TODO: check for 
    # use the first band as vza and vaa
    vza = vza[0]
    vaa = vaa[0]

    ntie_rows, ntie_columns = sza.shape
    tie_rows    = np.int32(np.linspace(0, shp[0]-1, ntie_rows))              # tie resolution, with target values
    tie_columns = np.int32(np.linspace(0, shp[1]-1, ntie_columns))           # tie resolution, with target values
    ds = ds.assign_coords(tie_rows = tie_rows, tie_columns = tie_columns)

    # initialize the dask arrays
    x = xr.DataArray(np.arange(len(ds.x)), dims=('x'))#.chunk(chunks[0])
    y = xr.DataArray(np.arange(len(ds.y)), dims=('y'))#.chunk(chunks[1])
    for name, tie in [(n.sza, sza),(n.saa, saa),(n.vza, vza),(n.vaa, vaa)]:
        ds[str(name)+'_tie'] = xr.DataArray(tie, dims=dims)
        ds[str(name)] = interp(
            ds[str(name)+'_tie'], tie_rows=Linear(x), tie_columns=Linear(y)
        )
    
    return ds


def _read_xml_block(item: dict, dims: tuple) -> xr.DataArray:
    """Parse XML values block into a float32 DataArray."""
    return xr.DataArray([i.split() for i in item['Values_List']['VALUES']], 
                        dims=dims).astype('float32')


class _LATLON:
    """
    Array-like object for lazy computation of MSI latitude/longitude arrays.
    
    Computes geographic coordinates from UTM projection parameters stored in metadata.
    Supports dask-based lazy evaluation for memory efficiency.
    """
    def __init__(self, geocoding: dict, kind: Literal['lat', 'lon'], ds: xr.Dataset):
        """Initialize coordinate calculator from geocoding metadata."""
        self.kind = kind

        code = geocoding.get('HORIZONTAL_CS_CODE')
        self.proj = pyproj.Proj(code)

        # lookup position in the UTM grid
        resolution = str(ds.resolution) if ds.resolution else '10'
        for e in geocoding.get('Geoposition'):
            if e['attributes']['resolution'] == resolution:
                ULX = e.get('ULX')
                ULY = e.get('ULY')
                XDIM = e.get('XDIM')
                YDIM = e.get('YDIM')

        assert (XDIM%2 == 0) and (YDIM%2 == 0)
        self.x = ULX + XDIM//2 + XDIM*da.arange(ds.totalheight)
        self.y = ULY + YDIM//2 + YDIM*da.arange(ds.totalwidth)

        self.shape = (ds.totalheight, ds.totalwidth)
        self.ndim = 2
        self.dtype = 'float32'

    def __getitem__(self, key):
        
        X, Y = self.x[key[1]], self.y[key[0]]
        if isinstance(key[0], slice) and isinstance(key[1], slice):
            # keys are both slices
            X, Y = da.meshgrid(X, Y)
        else:
            X, Y = da.broadcast_arrays(X, Y)

        lon, lat = self.proj(X, Y, inverse=True)

        if self.kind == 'lat': return lat.astype(self.dtype)
        else: return lon.astype(self.dtype)


def Level2_MSI(dirname: Union[str, Path]) -> xr.Dataset:
    """
    Read a Sentinel-2 MSI Level2A product as an xarray.Dataset.
    
    Note: This function is not yet implemented.
    
    Args:
        dirname: Path to the Sentinel-2 Level2A .SAFE directory
        
    Raises:
        NotImplementedError: Always raised as Level2A reading is not yet supported
    """
    raise NotImplementedError


def get_sample(level: int = 1) -> Path:
    """
    Download or retrieve a sample Sentinel-2 MSI product for testing.
    
    Requires the 'sand' module for Copernicus Data Space access.

    Args:
        level: Processing level (1 for Level1C, 2 for Level2A)

    Returns:
        Path to the downloaded .SAFE directory
        
    Raises:
        ImportError: If the 'sand' module is not installed
        
    Example:
        >>> safe_dir = get_sample(level=1)
        >>> ds = Level1_MSI(safe_dir)
    """
    try: 
        from sand.copernicus_dataspace import DownloadCDSE
        from sand.sample_product import products
    except ImportError:
        raise ImportError('To use get_sample function, you need to install SAND module')
    
    sensor = 'SENTINEL-2-MSI'
    prod_id = products[sensor][f'l{level}_product']
    dl = DownloadCDSE()
    target = dl.download_file(prod_id, env.getdir('DIR_SAMPLES'))
    assert target.exists()
    return target

def _v1_compat(ds: xr.Dataset) -> xr.Dataset:
    """Transform dataset to version 1 format for backward compatibility."""
    # Remove metadata
    ds.attrs['metadata_granule'] = filter_metadata(ds.attrs['metadata_granule'], [])
    ds.attrs['metadata'] = filter_metadata(ds.attrs['metadata'], [])
    
    # Apply previous rounded central wavelengths
    msi_band = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]
    ds = ds.assign_coords(bands=msi_band)
    
    # rename wavelength variable
    ds = ds.rename({str(n.cwav):'wav'})
    
    # add flags
    from core.tools import raiseflag
    ds[str(n.flags)] = xr.zeros_like(
        ds.vza,
        dtype=n.flags.dtype)
    raiseflag(
        ds[str(n.flags)],
        'L1_INVALID', 4,
        np.isnan(ds.vza)
    )
    
    return drop_unused_dims(ds)