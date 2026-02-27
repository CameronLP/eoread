#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np

from pathlib import Path
from typing import Literal, Union
from datetime import datetime
from warnings import filterwarnings
from re import findall

from core import env, log
from core.geo import n
from core.tools import getflags
from core.table import read_xml

from .eo import init_Rtoa
from .common import Interpolator, DataArray_from_array
from eoread.utils import spatial_resample, filter_metadata


# To filter warning message raised by instrument_data reading
filterwarnings('ignore', message=".*Duplicate dimension.*")


def get_sample(level: int = 1) -> Path:
    """
    Download or retrieve a sample OLCI product for testing.
    
    Requires the 'sand' module for EUMETSAT Data Store access.

    Args:
        level: Processing level (1 for Level1, 2 for Level2)

    Returns:
        Path to the downloaded .SEN3 directory
        
    Raises:
        ImportError: If the 'sand' module is not installed
        
    Example:
        >>> sen3_dir = get_sample(level=1)
        >>> ds = Level1_OLCI(sen3_dir)
    """
    try: 
        from sand.eumdac import DownloadEumDAC
        from sand.sample_product import products
    except ImportError:
        raise ImportError('To use get_sample function, you need to install SAND module')
    
    sensor = 'SENTINEL-3-OLCI-FR'
    prod_id = products[sensor][f'l{level}_product']

    targetdir = env.getdir('DIR_SAMPLES')
    dl = DownloadEumDAC()
    target = dl.download_file(prod_id, targetdir)
    assert target.exists()
    return target


def Level1_OLCI(
        dirname: Union[str, Path], 
        chunks: Union[int, tuple] = 500,
        tie_param: bool = False,
        interp_angles: Literal['atan2', 'linear', 'legacy'] = 'linear',
        metadata_template: Union[list, None] = None, 
        v1_compat: bool = False,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a Sentinel-3 OLCI Level1 product as an xarray.Dataset.
    
    OLCI (Ocean and Land Colour Instrument) provides 21 spectral bands from
    400nm to 1020nm with 300m spatial resolution.

    Args:
        dirname: Path to the OLCI .SEN3 directory
        chunks: Size of chunks for spatial dimensions. If int, applies to both dimensions.
        interp_angles: Interpolation method for angles:
                      - 'linear': Linear interpolation for all angles
                      - 'atan2': Trigonometric interpolation (sin/cos then atan2)
                      - 'legacy': Backward compatible (nearest for azimuth, linear for zenith)
        metadata_template: List of metadata keys to include. If None, includes all metadata.
                          Use empty list [] for minimal metadata.
        v1_compat: If True, formats output to match version 1 structure
        
    Example:
        >>> ds = Level1_OLCI('S3A_OL_1_EFR____*.SEN3/', chunks=1000)
    """
    ds = xr.Dataset()
    dirname = Path(dirname)
    if (dirname/dirname.name).exists(): dirname = (dirname/dirname.name)
    if isinstance(chunks, int): chunks = [chunks]*2
    chunks = dict(rows=chunks[0], columns=chunks[1])

    # read manifest file for file names and footprint
    if verbose: log.debug('Read metadata')
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    manifest = read_xml(dirname/'xfdumanifest.xml')
    ds.attrs['metadata'] = filter_fn(manifest, metadata_template)
    
    # Add latlon footprint
    footprint = manifest['metadataSection']['metadataObject'][2]
    footprint = footprint['metadataWrap']['xmlData']['frameSet']['footPrint']
    idata = iter(footprint['posList'].split())
    footprint = [(float(v), float(idata.__next__())) for v in idata]
    lat,lon = zip(*footprint)
    ds.attrs['footprint_lat'] = lat
    ds.attrs['footprint_lon'] = lon
    
    # Get band informations
    bandnames, cwvl = [], []
    info = manifest['metadataSection']['metadataObject'][4]
    info = info['metadataWrap']['xmlData']['olciProductInformation']
    for bn, data in info['bandDescriptions'].items():
        if bn == 'attributes': continue
        bandnames.append(bn)
        cwvl.append(data['centralWavelength'])
    ds = ds.assign_coords({str(n.bands): bandnames})
    ds = ds.assign({str(n.cwav): ((str(n.bands)),cwvl)})
    ds = ds.assign_coords({str(n.bgroup): (str(n.bands), ['bands_vnir']*len(cwvl))})
    
    # Check if product level is 1
    text = manifest['informationPackageMap']['contentUnit']['attributes']
    levels = findall(r'Level .', text['textInfo'])
    assert len(levels) == 1, f'Invalid textinfo in manifest: {text["textInfo"]}'
    level_from_manifest = levels[0].replace('Level ','level')
    assert 'level1' == level_from_manifest, \
        f'expected level1 encountered {level_from_manifest}'

    # Read main product
    if verbose: log.debug('Read radiances')
    ds = _read_bands(ds, dirname, chunks, 1)

    # Geo coordinates
    geo_coords_file = dirname/'geo_coordinates.nc'
    geo = xr.open_dataset(geo_coords_file, engine='h5netcdf').chunk(chunks)
    for k in geo.variables: 
        ds[k] = geo[k].astype('float32')
        ds[k].attrs.update(geo.attrs)

    # dimensions
    shape = ds.latitude.shape
    ac_factor = ds.latitude.ac_subsampling_factor
    al_factor = ds.latitude.al_subsampling_factor
    ds = ds.rename({'rows':str(n.rows), 'columns':str(n.columns)})

    # tie geometry interpolation
    if verbose: log.debug('read geometric tie points')
    tie_geom_file = dirname/'tie_geometries.nc'
    tie_ds = xr.open_dataset(tie_geom_file, engine='h5netcdf').chunk(chunks=-1)
    tie_ds = tie_ds.assign_coords(
        tie_columns=np.arange(tie_ds.sizes['tie_columns'])*ac_factor,
        tie_rows=np.arange(tie_ds.sizes['tie_rows'])*al_factor,
    )
    assert tie_ds.tie_columns[0] == ds[str(n.columns)][0]
    assert tie_ds.tie_columns[-1] == ds[str(n.columns)][-1]
    assert tie_ds.tie_rows[0] == ds[str(n.rows)][0]
    assert tie_ds.tie_rows[-1] == ds[str(n.rows)][-1]

    if interp_angles == 'linear': interp_aa, interp_za = 'linear', 'linear'
    elif interp_angles == 'atan2': interp_aa, interp_za = 'atan2', 'atan2'
    elif interp_angles == 'legacy': interp_aa, interp_za = 'nearest', 'linear'
    else: raise ValueError(f'Invalid interp_angles "{interp_angles}"')
    
    tie_chunks = tuple(chunks.values())
    shape = dict(tie_rows=shape[0], tie_columns=shape[1])
    for (ds_full, ds_tie, method) in [
                (str(n.sza), 'SZA', interp_za),
                (str(n.saa), 'SAA', interp_aa),
                (str(n.vza), 'OZA', interp_za),
                (str(n.vaa), 'OAA', interp_aa),
            ]:
        if method == 'atan2':
            _cos = spatial_resample(np.cos(np.radians(tie_ds[ds_tie])), shape, tie_chunks, 'linear')
            _sin = spatial_resample(np.sin(np.radians(tie_ds[ds_tie])), shape, tie_chunks, 'linear')
            ds[ds_full] = np.degrees(np.arctan2(_sin, _cos))
        else:
            ds[ds_full] = spatial_resample(tie_ds[ds_tie], shape, tie_chunks, method)
            pass
            
        ds[ds_full].attrs = tie_ds[ds_tie].attrs
        if tie_param: ds[ds_full+'_tie'] = tie_ds[ds_tie]

    # tie meteo interpolation
    if verbose: log.debug('read meteorological tie points')
    tie_meteo_file = dirname/'tie_meteo.nc'
    tie = xr.open_dataset(tie_meteo_file, engine='h5netcdf').chunk(chunks=-1)
    tie = tie.assign_coords(
                tie_columns = np.arange(tie.sizes['tie_columns'])*ac_factor,
                tie_rows = np.arange(tie.sizes['tie_rows'])*al_factor,
                )
    assert tie.tie_columns[0] == ds[str(n.columns)][0]
    assert tie.tie_columns[-1] == ds[str(n.columns)][-1]
    assert tie.tie_rows[0] == ds[str(n.rows)][0]
    assert tie.tie_rows[-1] == ds[str(n.rows)][-1]

    wind0 = spatial_resample(tie.horizontal_wind.isel(wind_vectors=0), shape, tie_chunks, 'linear')
    wind1 = spatial_resample(tie.horizontal_wind.isel(wind_vectors=1), shape, tie_chunks, 'linear')
    
    ds['horizontal_wind'] = np.sqrt(wind0**2 + wind1**2)
    ds['horizontal_wind'].attrs = tie['horizontal_wind'].attrs
    for var_from, var_to in [
        ('humidity', 'humidity'),
        ('sea_level_pressure', 'sea_level_pressure'),
        ('total_columnar_water_vapour', 'total_columnar_water_vapour'),
        ('total_ozone', 'total_column_ozone')
        ]:
        ds[var_to] = spatial_resample(tie[var_from], shape, tie_chunks, 'linear')
        ds[var_to].attrs = tie[var_from].attrs
        if tie_param: ds[var_to+'_tie'] = tie[var_from]

    # check subsampling factors
    assert ((ds.sizes[str(n.columns)]-1) == ac_factor*(tie_ds.sizes['tie_columns']-1))
    assert ((ds.sizes[str(n.rows)]-1) == al_factor*(tie_ds.sizes['tie_rows']-1))

    # instrument data
    instrument_data = xr.open_dataset(
        dirname/'instrument_data.nc',
        engine='h5netcdf',
        mask_and_scale=False,
        # this variable has duplicate dimensions, drop it
        drop_variables='relative_spectral_covariance'
    ).chunk(chunks=chunks)
    ds = ds.assign({x: instrument_data[x] for x in instrument_data.variables})

    # quality flags
    if verbose: log.debug('read quality masks')
    qf_file = dirname/'qualityFlags.nc'
    qf = xr.open_dataset(qf_file, engine='h5netcdf').chunk(chunks)
    for var in qf.variables: ds[var] = qf[var]
    
    # attributes
    if verbose: log.debug('add important attributes')
    meta = manifest['metadataSection']['metadataObject']
    date = meta[0]['metadataWrap']['xmlData']['acquisitionPeriod']
    start = datetime.fromisoformat(date['startTime'])
    stop  = datetime.fromisoformat(date['stopTime'])
    ds.attrs[str(n.datetime)] = (start + (stop - start)/2.).isoformat()
    
    platform = meta[1]['metadataWrap']['xmlData']['platform']
    ds.attrs[str(n.platform)] = platform['familyName'] + platform['number']
    ds.attrs[str(n.resolution)] = 500
    ds.attrs[str(n.sensor)] = platform['instrument']['familyName']['attributes']['abbreviation']
    ds.attrs[str(n.product_name)] = dirname.name
    ds.attrs[str(n.input_directory)] = str(dirname.parent)

    ds = ds.chunk(dict(detectors=-1))   # FIXME: do this upstream
    ds = ds.rename({'columns': str(n.columns), 'rows': str(n.rows)})
    
    if verbose: log.debug('compute reflectances')
    _olci_init_spectral(ds, chunks)
    ds = init_Rtoa(ds)

    if v1_compat: return _v1_compat(ds)
    return ds.unify_chunks()


def Level2_OLCI(
        dirname: Union[str, Path],
        chunks: Union[int, tuple] = 500,
        tie_param: bool = False,
        init_spectral: bool = True,
        interp_angles: Literal['atan2', 'linear', 'legacy'] = 'linear',
    ) -> xr.Dataset:
    """
    Read a Sentinel-3 OLCI Level2 product as an xarray.Dataset.
    
    Processes Level2 water products including water-leaving reflectances,
    chlorophyll concentration, aerosol properties, and quality flags.

    Args:
        dirname: Path to the OLCI Level2 .SEN3 directory
        chunks: Size of chunks for spatial dimensions
        tie_param: If True, keeps tie-point data in the output dataset
        init_spectral: If True, initializes spectral variables (wavelength, solar flux)
        interp_angles: Interpolation method for angles ('atan2', 'linear', or 'legacy')
    """
    ds = xr.Dataset()

    dirname = Path(dirname)
    if (dirname/dirname.name).exists():
        dirname = (dirname/dirname.name)

    # read manifest file for file names and footprint
    manifest = read_xml(dirname/'xfdumanifest.xml')
    ds.attrs.update(**manifest)
    
    # Add latlon footprint
    footprint = manifest['metadataSection']['metadataObject'][2]
    footprint = footprint['metadataWrap']['xmlData']['frameSet']['footPrint']
    idata = iter(footprint['posList'].split())
    footprint = [(float(v), float(idata.__next__())) for v in idata]
    lat,lon = zip(*footprint)
    ds.attrs['footprint_lat'] = lat
    ds.attrs['footprint_lon'] = lon
    
    # Get band informations
    bandnames, cwvl = [], []
    info = manifest['metadataSection']['metadataObject'][4]
    info = info['metadataWrap']['xmlData']['olciProductInformation']
    for bn, data in info['bandDescriptions'].items():
        if bn == 'attributes': continue
        bandnames.append(bn)
        cwvl.append(data['centralWavelength'])
    ds = ds.assign({str(n.cwav): (('bands'),cwvl)})
    ds = ds.assign({str(n.bnames): (('bands'),bandnames)})
    
    # Retrieve product level
    text = manifest['informationPackageMap']['contentUnit']['attributes']
    levels = findall(r'Level .', text['textInfo'])
    assert len(levels) == 1, f'Invalid textinfo in manifest: {text["textInfo"]}'
    level_from_manifest = levels[0].replace('Level ','level')
    assert 'level2' == level_from_manifest, \
        f'expected level2 encountered {level_from_manifest}'

    # Read main product
    ds = _read_bands(ds, dirname, chunks, 2)

    # Geo coordinates
    geo_coords_file = dirname/'geo_coordinates.nc'
    geo = xr.open_dataset(geo_coords_file, engine='h5netcdf').chunk(chunks=chunks)
    for k in geo.variables: 
        ds[k] = geo[k].astype('float32')
        ds[k].attrs.update(geo.attrs)

    # dimensions
    dims2 = ('rows','columns')
    shape2 = ds.latitude.shape
    ac_factor = ds.latitude.ac_subsampling_factor
    al_factor = ds.latitude.al_subsampling_factor

    # tie geometry interpolation
    tie_geom_file = dirname/'tie_geometries.nc'
    tie_ds = xr.open_dataset(tie_geom_file, engine='h5netcdf').chunk(chunks=-1)
    tie_ds = tie_ds.assign_coords(
        tie_columns=np.arange(tie_ds.sizes['tie_columns'])*ac_factor,
        tie_rows=np.arange(tie_ds.sizes['tie_rows'])*al_factor,
    )
    assert tie_ds.tie_columns[0] == ds.columns[0]
    assert tie_ds.tie_columns[-1] == ds.columns[-1]
    assert tie_ds.tie_rows[0] == ds.rows[0]
    assert tie_ds.tie_rows[-1] == ds.rows[-1]

    if interp_angles == 'linear': interp_aa, interp_za = 'linear', 'linear'
    elif interp_angles == 'atan2': interp_aa, interp_za = 'atan2', 'atan2'
    elif interp_angles == 'legacy': interp_aa, interp_za = 'nearest', 'linear'
    else: raise ValueError(f'Invalid interp_angles "{interp_angles}"')
    
    for (ds_full, ds_tie, method) in [
                ('sza', 'SZA', interp_za),
                ('saa', 'SAA', interp_aa),
                ('vza', 'OZA', interp_za),
                ('vaa', 'OAA', interp_aa),
            ]:
        if method == 'atan2':
            _cos = DataArray_from_array(
                Interpolator(shape2, np.cos(np.radians(tie_ds[ds_tie].astype('float32'))), 'linear'),
                dims2,
                chunks,
            )
            _sin = DataArray_from_array(
                Interpolator(shape2, np.sin(np.radians(tie_ds[ds_tie].astype('float32'))), 'linear'),
                dims2,
                chunks,
            )
            ds[ds_full] = np.degrees(np.arctan2(_sin, _cos))
        else:
            ds[ds_full] = DataArray_from_array(
                Interpolator(shape2, tie_ds[ds_tie].astype('float32'), method),
                dims2,
                chunks,
            )
        ds[ds_full].attrs = tie_ds[ds_tie].attrs
        if tie_param:
            ds[ds_full+'_tie'] = tie_ds[ds_tie]

    # tie meteo interpolation
    tie_meteo_file = dirname/'tie_meteo.nc'
    tie = xr.open_dataset(tie_meteo_file, engine='h5netcdf').chunk(chunks=-1)
    tie = tie.assign_coords(
                tie_columns = np.arange(tie.sizes['tie_columns'])*ac_factor,
                tie_rows = np.arange(tie.sizes['tie_rows'])*al_factor,
                )
    assert tie.tie_columns[0] == ds.columns[0]
    assert tie.tie_columns[-1] == ds.columns[-1]
    assert tie.tie_rows[0] == ds.rows[0]
    assert tie.tie_rows[-1] == ds.rows[-1]
    
    wind0 = DataArray_from_array(
        Interpolator(
            shape2,
            tie.horizontal_wind.isel(wind_vectors=0)
        ),
        dims2,
        chunks,
    )
    wind1 = DataArray_from_array(
        Interpolator(
            shape2,
            tie.horizontal_wind.isel(wind_vectors=1)
        ),
        dims2,
        chunks,
    )
    ds['horizontal_wind'] = np.sqrt(wind0**2 + wind1**2)
    ds['horizontal_wind'].attrs = tie['horizontal_wind'].attrs
    for var_from, var_to in [
        ('humidity', 'humidity'),
        ('sea_level_pressure', 'sea_level_pressure'),
        ('total_columnar_water_vapour', 'total_columnar_water_vapour'),
        ('total_ozone', 'total_column_ozone')
        ]:
        ds[var_to] = DataArray_from_array(
            Interpolator(shape2, tie[var_from]),
            dims2,
            chunks,
        )
        ds[var_to].attrs = tie[var_from].attrs
        if tie_param:
            ds[var_to+'_tie'] = tie[var_from]

    # check subsampling factors
    assert ((ds.sizes['columns']-1) == ac_factor*(tie_ds.sizes['tie_columns']-1))
    assert ((ds.sizes['rows']-1) == al_factor*(tie_ds.sizes['tie_rows']-1))

    # instrument data
    instrument_data_file = dirname/'instrument_data.nc'
    instrument_data = xr.open_dataset(instrument_data_file,
                                      engine='h5netcdf',
                                      mask_and_scale=False,
                                      # this variable has duplicate dimensions, drop it
                                      drop_variables='relative_spectral_covariance'
                                      ).chunk(chunks=chunks)
    for x in instrument_data.variables:
        ds[x] = instrument_data[x]

    # chl_nn
    fname = os.path.join(dirname, 'chl_nn.nc')
    qf = xr.open_dataset(fname, engine='h5netcdf').chunk(chunks=chunks)
    ds['chl_nn'] = qf.CHL_NN

    # chl_oc4me
    fname = os.path.join(dirname, 'chl_oc4me.nc')
    qf = xr.open_dataset(fname, engine='h5netcdf').chunk(chunks=chunks)
    ds['chl_oc4me'] = qf.CHL_OC4ME

    # quality flags
    fname = os.path.join(dirname, 'wqsf.nc')
    qf = xr.open_dataset(fname, engine='h5netcdf').chunk(chunks=chunks)
    ds['wqsf'] = qf.WQSF

    # aerosol properties
    fname = os.path.join(dirname, 'w_aer.nc')
    qf = xr.open_dataset(fname, engine='h5netcdf').chunk(chunks=chunks)
    ds['A865'] = qf.A865
    ds['T865'] = qf.T865

    # flags
    # if level == 'level1':
        # ds[naming.flags] = xr.zeros_like(
        #     ds.vza,
        #     dtype=naming.flags_dtype)
        # qf = getflags(ds.quality_flags)

        # # raise LAND mask when land is raised but not fresh_inland_water
        # raiseflag(
        #     ds[naming.flags],
        #     "LAND",
        #     flags["LAND"],
        #     ds.quality_flags & (qf["land"] + qf["fresh_inland_water"]) == qf["land"],
        # )
        # raiseflag(
        #     ds[naming.flags],
        #     "L1_INVALID",
        #     flags["L1_INVALID"],
        #     ds.quality_flags & qf["invalid"],
        # )
    
    # attributes
    # ds.attrs[naming.datetime] = (dstart + (dstop - dstart)/2.).isoformat()
    ds.attrs[str(n.platform)] = 'Sentinel-3'   # FIXME: A or B
    ds.attrs[str(n.sensor)] = 'OLCI'
    ds.attrs[str(n.input_directory)] = os.path.dirname(dirname)

    ds = ds.chunk(dict(detectors=-1))   # FIXME: do this upstream

    if init_spectral: _olci_init_spectral(ds, chunks)

    ds = ds.rename({'columns': str(n.columns), 'rows': str(n.rows)})

    return ds.unify_chunks()


def get_sample(level: int = 1) -> Path:
    """
    Download or retrieve a sample OLCI product for testing.
    
    Requires the 'sand' module for EUMETSAT Data Store access.

    Args:
        level: Processing level (1 for Level1, 2 for Level2)

    Returns:
        Path to the downloaded .SEN3 directory
        
    Raises:
        ImportError: If the 'sand' module is not installed
    """
    return collect_sample(f'LEVEL{level}_OLCI', 'eumdac', 'SENTINEL-3-OLCI-FR', level)



################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def read_ltao(dirname: Path, band: str, chunks: dict, collec: dict) -> None:
        """Read Level1 radiance data for a single band."""
        filename = dirname/f'{band}_radiance.nc'
        data = xr.open_dataarray(filename, engine='h5netcdf').chunk(chunks)
        prod_list.append(data)

    if level == 1: 
        param_name, unit = str(n.ltoa), 'W/sr/m^2'
    else: 
        param_name, unit = str(n.rho_w), None
    
    ds[param_name] = xr.concat(prod_list, dim=str(n.bands))
    ds[param_name].attrs.update(unit=unit)
    return ds

def _olci_init_spectral(ds: xr.Dataset, chunks: dict) -> None:
    """
    Broadcast spectral (detector-wise) data to the full image grid.
    
    Extracts detector-specific wavelengths and solar fluxes and maps them
    to each pixel based on the detector_index variable.
    
    Adds the following variables to ds (in place):
        - wav: Wavelength for each pixel and band (nm)
        - F0: Solar flux for each pixel and band
    """
    # wavelength
    ds[str(n.wav)] = xr.apply_ufunc(
        lambda l0, di: l0[:,0,0,di],
        ds.lambda0,  # (bands x detectors)
        ds.detector_index,   # (rows x columns)
        dask='parallelized',
        input_core_dims=[['detectors'], []],
        output_dtypes=[ds.lambda0.dtype],
    )
    ds[str(n.wav)].attrs.update(ds.lambda0.attrs)

    # solar flux
    ds[str(n.F0)] = xr.apply_ufunc(
        lambda sf, di: sf[:,0,0,di],
        ds.solar_flux,  # (bands x detectors)
        ds.detector_index,   # (rows x columns)
        dask='parallelized',
        input_core_dims=[['detectors'], []],
        output_dtypes=[ds.solar_flux.dtype],
    )
    ds[str(n.F0)].attrs.update(ds.solar_flux.attrs)


def _decompose_flags(value: int, flags: dict) -> list:
    """Return list of flag meanings for a given binary flag value."""
    return [m for (m, v) in flags.items() if (v & value != 0)]


def get_valid_l2_pixels(
        wqsf: xr.DataArray, 
        flags: list = [
            'INVALID', 'LAND', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN',
            'SNOW_ICE', 'SUSPECT', 'HISOLZEN', 'SATURATED', 'HIGHGLINT', 'WHITECAPS',
            'AC_FAIL', 'OC4ME_FAIL', 'ANNOT_TAU06', 'RWNEG_O2', 'RWNEG_O3', 'RWNEG_O4',
            'RWNEG_O5', 'RWNEG_O6', 'RWNEG_O7', 'RWNEG_O8', 'ANNOT_ABSO_D',
            'ANNOT_DROUT', 'ANNOT_MIXR1']
    ) -> xr.DataArray:
    """
    Get valid Level2 water pixels by masking specified quality flags.
    
    Args:
        wqsf: Water Quality and Science Flags array
        flags: List of flag names to mask out. Pixels with any of these flags
              raised will be marked as invalid.
    
    Returns:
        Boolean array where True indicates valid pixels
    """
    bval = 0
    L2_FLAGS = getflags(wqsf)
    for flag in flags:
        bval += int(L2_FLAGS[flag])

    return wqsf & bval == 0


def _v1_compat(ds: xr.Dataset) -> xr.Dataset:
    """Transform dataset to version 1 format for backward compatibility."""
    # Reset band coordinates
    ds = ds.assign_coords(bands=[400, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709, 754, 760, 764, 767, 779, 865, 885, 900, 940, 1020]) 
    
    # rename bands variable
    ds = ds.assign({str(n.rtoa): ((str(n.bands), str(n.rows), str(n.columns)), ds[str(n.rtoa)].data)})
    
    # Add flags
    ds[str(n.flags)] = xr.zeros_like(
        ds.vza,
        dtype=n.flags.dtype)
    qf = getflags(ds.quality_flags)

    # raise LAND mask when land is raised but not fresh_inland_water
    from .eo import raiseflag
    raiseflag(
        ds[str(n.flags)],
        "LAND", 1,
        ds.quality_flags & (qf["land"] + qf["fresh_inland_water"]) == qf["land"],
    )
    raiseflag(
        ds[str(n.flags)],
        "L1_INVALID", 4,
        ds.quality_flags & qf["invalid"],
    )
    
    # Complete attributes
    for k,v in list(ds.longitude.attrs.items())[5:]: ds.attrs[k] = v
    
    return ds