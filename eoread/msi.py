#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Update processing baseline 4.00
# https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-upcoming

from pyproj import Proj
from pathlib import Path
from typing import Literal, Union

import numpy as np
import xarray as xr

from core.interpolate import interp, Linear
from core.tools import merge, drop_unused_dims, only
from core.geo.naming import names
from core.table import read_xml
from core import env, log

from eoread.flags import FlagsReaderBase, GenericFlags
from eoread.tools import (
    filter_metadata, 
    spatial_resample, 
    format_chunks, 
    collect_sample
)


user_guide = 'https://sentinels.copernicus.eu/documents/247904/685211/Sentinel-2_User_Handbook.pdf/8869acdf-fd84-43ec-ae8c-3e80a436a16c?t=1438278087000'

def Level1_MSI(
        dirname: Union[str, Path],
        chunks: Union[int, tuple, dict] = 500,
        resolution: Literal[10,20,60,None] = 60,
        metadata_template: Union[list, None] = None, 
        read_mask: bool = False,
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
    
    # Check that folder exists
    ds = xr.Dataset()
    dirname = Path(dirname).resolve()
    assert dirname.exists(), 'Folder does not exist'
    
    # Format chunks
    chunks = format_chunks(chunks)

    if list(dirname.glob('GRANULE')):
        granules = list((dirname/'GRANULE').glob('*'))
        assert len(granules) == 1
        granule_dir = granules[0]
    else: 
        granule_dir = dirname

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

    # add several attributes
    if verbose: log.debug('Add important attributes')
    sensing_time = xmlgranule['General_Info']['SENSING_TIME']['values']
    ds.attrs[str(names.datetime)] = sensing_time
    ds.attrs[str(names.platform)] = {
        "S2A": "Sentinel-2A",
        "S2B": "Sentinel-2B",
        "S2C": "Sentinel-2C",
    }[platform]
    ds.attrs[str(names.resolution)] = resolution
    ds.attrs[str(names.sensor)] = 'MSI'
    ds.attrs[str(names.product_name)] = dirname.name
    ds.attrs[str(names.input_directory)] = str(dirname.parent)
    ds.attrs['user_guide'] = user_guide

    # msi_read_toa and quality masks
    if verbose: log.debug('Read top of atmosphere data')
    ds = _Internal.read_toa(ds, granule_dir, processing_baseline, product_image, chunks, metadata, resolution)

    # Compute latitude and longitude
    if verbose: log.debug('Generate latlon rasters')
    ds = _Internal.supplement_latlon(ds, chunks, xmlgranule)

    # msi_read_geometry
    if verbose: log.debug('Read geometric angles')
    tileangles = xmlgranule['Geometric_Info']['Tile_Angles']
    ds = _Internal.read_geometry(ds, tileangles, resolution, chunks)
    
    # msi read quality mask
    if read_mask:
        if verbose: log.debug('Read quality masks')
        ds = _Internal.read_qi(ds, granule_dir, chunks)
    else:
        if verbose: log.debug('skip reading of quality masks')
    
    # Assign new coordinates for band and wavelength variable
    ds = ds.assign({str(names.cwav): ((str(names.bands)), metadata['cwvl'])})
    ds = ds.assign_coords({str(names.bands): metadata['name']})
    ds = _Internal.update_spatial_coords(ds)
    
    # Filter metadata
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs['metadata_granule'] = filter_fn(xmlgranule, metadata_template)
    ds.attrs['metadata'] = filter_fn(xmlroot, metadata_template)
    ds.attrs['_flag_reader'] = 'eoread.msi.FlagsReader_MSI'

    # SRF getter
    ds.attrs['_srf_getter'] = 'eotools.srf.get_SRF_eumetsat'
    ds.attrs['_srf_getter_arg'] = {
        'S2A': 'sentinel2_1_msi', 
        'S2B': 'sentinel2_2_msi', 
        'S2C': 'sentinel2_3_msi', 
    }[platform]
    
    ds = drop_unused_dims(ds)
    if resolution:
        ds = ds.transpose(..., *[str(names.rows), str(names.columns)])
        ds = ds.set_coords(str(names.bgroup))
    
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
    """
    return collect_sample(f'LEVEL{level}_MSI', 'cdse', 'SENTINEL-2-MSI', level)


class FlagsReader_MSI(FlagsReaderBase):
    """
    Flags reader for MSI (Sentinel-2) data.
    
    Since MSI L1C products don't currently read quality masks in eoread,
    this flags reader determines flags based on data validity.
    """
    
    def requires(self) -> list[str]:
        """Variables required for flag determination."""
        return [str(names.vza)]  # Use viewing zenith angle as reference
    
    def dims_like(self) -> str:
        """Returns a variable name with the same shape as the output."""
        return str(names.vza)
    
    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        """
        Retrieve a specific quality flag from the MSI dataset.
        
        Args:
            ds: MSI dataset
            flag_name: Standard flag identifier (currently only L1_INVALID supported)
            
        Returns:
            Boolean DataArray indicating invalid pixels (True where VZA is NaN)
            
        Raises:
            ValueError: If the requested flag type is not supported
        """
        if flag_name == GenericFlags.L1_INVALID:
            # L1_INVALID is True where vza is NaN (invalid data)
            return xr.DataArray(
                np.isnan(ds[str(names.vza)]),
                dims=ds[str(names.vza)].dims,
                coords=ds[str(names.vza)].coords
            )
        else:
            raise ValueError(f"Unsupported flag: {flag_name}")
    
    def getflag_raw(self, ds: xr.Dataset, flag_name: str) -> xr.DataArray:
        """Get a raw flag (not implemented for MSI since no quality masks are read)."""
        raise NotImplementedError("MSI does not currently read raw quality flags")


################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def supplement_latlon(ds: xr.Dataset, chunks: list, xmlgranule: dict) -> xr.Dataset:
        """Generate latitude and longitude arrays from UTM projection metadata."""
        
        # Get tile projection from metadata
        geocoding = xmlgranule['Geometric_Info']['Tile_Geocoding']
        code = geocoding.get('HORIZONTAL_CS_CODE')
        proj = Proj(code)
        
        # lookup position in the UTM grid
        resolution = str(ds.resolution) if ds.resolution else '10'
        for e in geocoding.get('Geoposition'):
            if e['attributes']['resolution'] == resolution:
                ULX = e.get('ULX')
                ULY = e.get('ULY')
                XDIM = e.get('XDIM')
                YDIM = e.get('YDIM')
                break
        
        # Compute latitude and longitude rasters
        assert (XDIM%2 == 0) and (YDIM%2 == 0)
        x = ULX + XDIM//2 + XDIM * np.arange(ds.totalheight)
        y = ULY + YDIM//2 + YDIM * np.arange(ds.totalwidth)
        X, Y = np.meshgrid(x, y)
        lon, lat = proj(X, Y, inverse=True)
        
        # Add rasters in the dataset
        dims = [str(names.rows), str(names.columns)]
        return ds.assign({
            str(names.lon): xr.DataArray(lon, dims=dims).chunk(chunks), 
            str(names.lat): xr.DataArray(lat, dims=dims).chunk(chunks)
        })
            
    @staticmethod
    def read_xml_block(item: dict, dims: tuple) -> xr.DataArray:
        """Parse XML values block into a float32 DataArray."""
        values = [i.split() for i in item['Values_List']['VALUES']]
        return xr.DataArray(values, dims=dims).astype('float32')

    @staticmethod
    def read_qi(ds: xr.Dataset, granule_dir: Path, chunks: list) -> xr.Dataset:
        """Read quality indicator masks from QI_DATA directory."""
        for filename in (granule_dir/'QI_DATA').glob(f'*.jp2'):
            
            if '_PVI' in filename.stem: continue
            arr = xr.open_dataarray(filename, engine='rasterio')
            arr = arr.astype(arr.encoding['dtype'])
            arr = arr.rename(x='x_red', y='y_red')
            ds[filename.stem] = arr.rename({'band': str(names.detector)})
        
        ds = ds.rename_vars({'MSK_CLASSI_B00':'MSK_CLASSI'})
        ds = merge(ds, dim=str(names.bands), pattern=r'(.+)_B(.+)', dtype=str)

        return ds
    
    @staticmethod
    def read_boa(
            ds: xr.Dataset, 
            granule_dir: Path, 
            processing_baseline: str, 
            product_image: dict, 
            chunks: dict, 
            metadata: dict, 
            resolution: int
        ) -> xr.Dataset:
        """Read and calibrate TOA reflectance from JP2 files.
        
        Applies radiometric offset correction (baseline >= 4.0) and quantification.
        Optionally resamples all bands to 10m resolution if concat=True.
        """
        
        # Retrieve radiometric offset
        if float(processing_baseline) >= 4:
            offset = product_image['BOA_ADD_OFFSET_VALUES_LIST']['BOA_ADD_OFFSET']
            radio_offset = [int(x['values']) for x in offset]
        else: 
            radio_offset = [0]*len(metadata['name'])
        
        # Open deserved bands
        reso = ds.resolution if ds.resolution else 60
        directory = granule_dir/'IMG_DATA'/f'R{reso}m'
        quantif = product_image['QUANTIFICATION_VALUES_LIST']
        quantif = quantif['BOA_QUANTIFICATION_VALUE']['values']
        for iband, bname in enumerate(metadata['name']):
            
            # Bands 8 and 10 are not in level 2 product
            if bname in ['B8','B10']: continue
            
            # Retrieve filename
            bname_ = bname.replace('B', 'B0') if len(bname)==2 else bname
            filename = only(directory.glob(f'*_{bname_}_*.jp2'))
            
            # Open band and compute reflectance
            array = xr.open_dataarray(filename, engine='rasterio').squeeze()
            array = array.rename(x=str(names.columns), y=str(names.rows)).chunk(chunks)
            array = ((array+radio_offset[iband])/quantif)
            
            # Resample the array and add it to the dataset
            if resolution:
                shape = {str(names.columns): ds.totalwidth, str(names.rows): ds.totalheight} 
                resampled = spatial_resample(array, shape, chunks)
                ds[str(names.rtoa)+f'_{bname}'] = resampled
            else:
                res = str(metadata['resolution'][iband]) + 'm'
                name = str(names.rtoa)+f'_{res}_{bname}'
                array = array.squeeze().chunk(chunks)
                ds[name] = array.rename(
                    y=f'{str(names.rows)}_{res}', 
                    x=f'{str(names.columns)}_{res}'
                )
        
        # Merge reflectance bands
        if resolution:
            res = [names.bands+f'_{r}m' for r in metadata['resolution']]
            res.pop(7); res.pop(10) # drop band 8 and 10
            ds = merge(ds, dim=str(names.bands), pattern=r'(.+)_B(.+)', dtype=str)
            ds[str(names.rtoa)].attrs.update(unit=None)
            ds = ds.assign({str(names.bgroup): (str(names.bands), res)})
        else:
            ds = merge(ds, dim=str(names.bands+'_10m'), pattern=r'(.+_10m)_B(.+)', dtype=str)
            ds = merge(ds, dim=str(names.bands+'_20m'), pattern=r'(.+_20m)_B(.+)', dtype=str)
            ds = merge(ds, dim=str(names.bands+'_60m'), pattern=r'(.+_60m)_B(.+)', dtype=str)
            ds[str(names.rtoa)+'_10m'].attrs.update(unit=None)
            ds[str(names.rtoa)+'_20m'].attrs.update(unit=None)
            ds[str(names.rtoa)+'_60m'].attrs.update(unit=None)
        
        return ds
    
    @staticmethod
    def read_toa(
            ds: xr.Dataset, 
            granule_dir: Path, 
            processing_baseline: str, 
            product_image: dict, 
            chunks: dict, 
            metadata: dict, 
            resolution: int
        ) -> xr.Dataset:
        """Read and calibrate TOA reflectance from JP2 files.
        
        Applies radiometric offset correction (baseline >= 4.0) and quantification.
        Optionally resamples all bands to 10m resolution if concat=True.
        """
        
        # Retrieve radiometric offset
        if float(processing_baseline) >= 4:
            offset = product_image['Radiometric_Offset_List']['RADIO_ADD_OFFSET']
            radio_offset = [int(x['values']) for x in offset]
        else: 
            radio_offset = [0]*len(metadata['name'])
        
        # Open deserved bands
        quantif = product_image['QUANTIFICATION_VALUE']['values']
        for iband, bname in enumerate(metadata['name']):
            
            # Retrieve filename
            bname_ = bname.replace('B', 'B0') if len(bname)==2 else bname
            filename = only((granule_dir/'IMG_DATA').glob(f'*_{bname_}.jp2'))
            
            # Open band and compute reflectance
            array = xr.open_dataarray(filename, engine='rasterio').squeeze()
            array = array.rename(x=str(names.columns), y=str(names.rows)).chunk(chunks)
            array = ((array+radio_offset[iband])/quantif)
            
            # Resample the array and add it to the dataset
            if resolution:
                shape = {str(names.columns): ds.totalwidth, str(names.rows): ds.totalheight} 
                resampled = spatial_resample(array, shape, chunks)
                ds[str(names.rtoa)+f'_{bname}'] = resampled
            else:
                res = str(metadata['resolution'][iband]) + 'm'
                name = str(names.rtoa)+f'_{res}_{bname}'
                array = array.squeeze().chunk(chunks)
                ds[name] = array.rename(
                    y=f'{str(names.rows)}_{res}', 
                    x=f'{str(names.columns)}_{res}'
                )
        
        # Merge reflectance bands
        if resolution:
            res = [names.bands+f'_{r}m' for r in metadata['resolution']]
            ds = merge(ds, dim=str(names.bands), pattern=r'(.+)_B(.+)', dtype=str)
            ds[str(names.rtoa)].attrs.update(unit=None)
            ds = ds.assign({str(names.bgroup): (str(names.bands), res)})
        else:
            ds = merge(ds, dim=str(names.bands+'_10m'), pattern=r'(.+_10m)_B(.+)', dtype=str)
            ds = merge(ds, dim=str(names.bands+'_20m'), pattern=r'(.+_20m)_B(.+)', dtype=str)
            ds = merge(ds, dim=str(names.bands+'_60m'), pattern=r'(.+_60m)_B(.+)', dtype=str)
            ds[str(names.rtoa)+'_10m'].attrs.update(unit=None)
            ds[str(names.rtoa)+'_20m'].attrs.update(unit=None)
            ds[str(names.rtoa)+'_60m'].attrs.update(unit=None)
        
        return ds
    
    @staticmethod
    def read_geometry(
            ds: xr.Dataset, 
            tileangles: dict, 
            resolution: str,
            chunks: dict
        ) -> xr.Dataset:
        """Read and interpolate geometric angles from tie points to full resolution."""
        
        # read solar angles at tiepoints
        dims = ('tie_rows', 'tie_columns')
        sza = _Internal.read_xml_block(tileangles['Sun_Angles_Grid']['Zenith'], dims)
        saa = _Internal.read_xml_block(tileangles['Sun_Angles_Grid']['Azimuth'], dims)
        shape = (ds.totalheight, ds.totalwidth)

        # read view angles (for each band)
        vza, vaa = {}, {}
        tie_shape = sza.shape
        for e in tileangles.get('Viewing_Incidence_Angles_Grids'):

            bandid = int(e['attributes']['bandId'])
            for name, angle in [('Zenith', vza), ('Azimuth', vaa)]: 
                
                # Reading view angles
                data = _Internal.read_xml_block(e[name], dims)
                data = data.values.flatten()
                
                # Add data in dictionary by composition
                if bandid not in angle: 
                    angle[bandid] = data
                else:
                    valid = ~np.isnan(data) # indexes where the data is not null
                    angle[bandid][valid] = data[valid]

        # reshape to original
        vza = {b: v.reshape(tie_shape) for b,v in vza.items()}
        vaa = {b: v.reshape(tie_shape) for b,v in vaa.items()}

        # use the first band as vza and vaa
        vza = vza[0]
        vaa = vaa[0]
        
        # Assign coordinates for tie points dimensions
        tie_rows = np.int32(np.linspace(0, shape[0]-1, tie_shape[0]))
        tie_columns = np.int32(np.linspace(0, shape[1]-1, tie_shape[1]))
        ds = ds.assign_coords(tie_rows=tie_rows, tie_columns=tie_columns)
        
        # Determine dimension names
        if resolution:
            d = dict(x=str(names.columns), y=str(names.rows)) 
        else:
            d = dict(x=f'{str(names.columns)}_60m', y=f'{str(names.rows)}_60m')
        
        # Initialize the dask arrays as new coordinates
        x = np.linspace(0, ds.tie_columns[-1].values, len(ds[d['x']]))
        x = xr.DataArray(x, dims=(d['x'])).chunk(chunks[str(names.columns)])
        y = np.linspace(0, ds.tie_rows[-1].values, len(ds[d['y']]))
        y = xr.DataArray(y, dims=(d['y'])).chunk(chunks[str(names.rows)])
        
        # Interpolate angle rasters
        angles = [(names.sza, sza), (names.saa, saa), (names.vza, vza), (names.vaa, vaa)]
        for name, tie in angles:
            ds[str(name)+'_tie'] = xr.DataArray(tie, dims=dims)
            ds[str(name)] = interp(
                ds[str(name)+'_tie'], tie_rows=Linear(x), tie_columns=Linear(y)
            )
        
        return ds
    
    @staticmethod
    def update_spatial_coords(ds: xr.Dataset) -> xr.Dataset:
        """Update spatial dimension coordinates to integer ranges."""
        
        # Define list of possible spatial dimensions
        dimensions = [str(names.rows), str(names.columns)]
        dimensions += [str(names.rows) + f'_{res}m' for res in [10,20,60]]
        dimensions += [str(names.columns) + f'_{res}m' for res in [10,20,60]]        
        
        # Update every spatial dimension as a range
        for dim in ds.dims:
            if dim == str(names.rows) or dim == str(names.columns):
                ds = ds.assign_coords({dim: np.arange(len(ds[dim]))})
            
        return ds