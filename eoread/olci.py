#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import dask.array as da
from pathlib import Path
from typing import Literal
from datetime import datetime
from re import findall

from core import env, log
from core.geo import n
from core.tools import getflags
from core.table import read_xml

from .eo import init_Rtoa
from .common import Interpolator, DataArray_from_array
from eoread.utils import spatial_resample, filter_metadata


def get_sample(level:int=1, use_cache:bool=True) -> Path:
    try: 
        from core.files.cache import cache_dataframe
        from sand.copernicus_dataspace import DownloadCDSE
        from sand.sample_product import products
    except ImportError:
        log.error('To use get_sample function, you need to install SAND module',
                  e=ImportError)
        
    cachefile = env.getdir('DIR_STATIC')/'query_olci.pickle'
    if use_cache: cache_deco = cache_dataframe(cachefile)
    else: cache_deco = lambda x: x
    
    sensor = 'SENTINEL-3-OLCI-FR'
    params = products[sensor][f'level{level}']
    dl = DownloadCDSE(sensor, level)
    ls = cache_deco(dl.query)(**params)
    return dl.download(ls.iloc[0], env.getdir('DIR_SAMPLES'))


def Level1_OLCI(dirname, 
                chunks: int|tuple = 500,
                tie_param: bool = False,
                interp_angles: Literal['atan2','linear','legacy'] = 'linear',
                metadata_template: list = None, 
                v1_compat: bool = False):
    '''
    Read an OLCI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances, reflectances, the angles on the full grid, etc.
    
    interp_angles:
        'linear': linear interpolation
        'atan2': interpolate sin(x) and cos(x), then x = atan2(sin, cos)
        'legacy': for backward compatibility (nearest for azimuth angles, linear for zenith angles)
    '''
    ds = xr.Dataset()
    dirname = Path(dirname)
    if (dirname/dirname.name).exists(): dirname = (dirname/dirname.name)
    if isinstance(chunks, int): chunks = [chunks]*2
    chunks = dict(rows=chunks[0], columns=chunks[1])

    # read manifest file for file names and footprint
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
    ds = ds.assign({n.cwav.name: ((n.bands.name),cwvl)})
    ds = ds.assign({n.bnames.name: ((n.bands.name),bandnames)})
    
    # Check if product level is 1
    text = manifest['informationPackageMap']['contentUnit']['attributes']
    levels = findall(r'Level .', text['textInfo'])
    assert len(levels) == 1, f'Invalid textinfo in manifest: "{text['textInfo']}"'
    level_from_manifest = levels[0].replace('Level ','level')
    assert 'level1' == level_from_manifest, \
        f'expected level1 encountered {level_from_manifest}'

    # Read main product
    ds = read_bands(ds, dirname, chunks, 1)

    # Geo coordinates
    geo_coords_file = dirname/'geo_coordinates.nc'
    geo = xr.open_dataset(geo_coords_file).chunk(chunks)
    for k in geo.variables: 
        ds[k] = geo[k].astype('float32')
        ds[k].attrs.update(geo.attrs)

    # dimensions
    shape = ds.latitude.shape
    ac_factor = ds.latitude.ac_subsampling_factor
    al_factor = ds.latitude.al_subsampling_factor
    ds = ds.rename({'rows':n.rows.name, 'columns':n.columns.name})

    # tie geometry interpolation
    tie_geom_file = dirname/'tie_geometries.nc'
    tie_ds = xr.open_dataset(tie_geom_file).chunk(chunks=-1)
    tie_ds = tie_ds.assign_coords(
        tie_columns=da.arange(tie_ds.sizes['tie_columns'])*ac_factor,
        tie_rows=da.arange(tie_ds.sizes['tie_rows'])*al_factor,
    )
    assert tie_ds.tie_columns[0] == ds[n.columns.name][0]
    assert tie_ds.tie_columns[-1] == ds[n.columns.name][-1]
    assert tie_ds.tie_rows[0] == ds[n.rows.name][0]
    assert tie_ds.tie_rows[-1] == ds[n.rows.name][-1]

    if interp_angles == 'linear': interp_aa, interp_za = 'linear', 'linear'
    elif interp_angles == 'atan2': interp_aa, interp_za = 'atan2', 'atan2'
    elif interp_angles == 'legacy': interp_aa, interp_za = 'nearest', 'linear'
    else: raise ValueError(f'Invalid interp_angles "{interp_angles}"')
    
    tie_chunks = tuple(chunks.values())
    shape = dict(tie_rows=shape[0], tie_columns=shape[1])
    for (ds_full, ds_tie, method) in [
                (n.sza.name, 'SZA', interp_za),
                (n.saa.name, 'SAA', interp_aa),
                (n.vza.name, 'OZA', interp_za),
                (n.vaa.name, 'OAA', interp_aa),
            ]:
        if method == 'atan2':
            _cos = spatial_resample(da.cos(da.radians(tie_ds[ds_tie])), shape, tie_chunks, 'linear')
            _sin = spatial_resample(da.sin(da.radians(tie_ds[ds_tie])), shape, tie_chunks, 'linear')
            ds[ds_full] = da.degrees(da.arctan2(_sin, _cos))
        else:
            ds[ds_full] = spatial_resample(tie_ds[ds_tie], shape, tie_chunks, method)
            
        ds[ds_full].attrs = tie_ds[ds_tie].attrs
        if tie_param: ds[ds_full+'_tie'] = tie_ds[ds_tie]

    # tie meteo interpolation
    tie_meteo_file = dirname/'tie_meteo.nc'
    tie = xr.open_dataset(tie_meteo_file).chunk(chunks=-1)
    tie = tie.assign_coords(
                tie_columns = da.arange(tie.sizes['tie_columns'])*ac_factor,
                tie_rows = da.arange(tie.sizes['tie_rows'])*al_factor,
                )
    assert tie.tie_columns[0] == ds[n.columns.name][0]
    assert tie.tie_columns[-1] == ds[n.columns.name][-1]
    assert tie.tie_rows[0] == ds[n.rows.name][0]
    assert tie.tie_rows[-1] == ds[n.rows.name][-1]

    wind0 = spatial_resample(tie.horizontal_wind.isel(wind_vectors=0), shape, tie_chunks, 'linear')
    wind1 = spatial_resample(tie.horizontal_wind.isel(wind_vectors=1), shape, tie_chunks, 'linear')
    
    ds['horizontal_wind'] = da.sqrt(wind0**2 + wind1**2)
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
    assert ((ds.sizes[n.columns.name]-1) == ac_factor*(tie_ds.sizes['tie_columns']-1))
    assert ((ds.sizes[n.rows.name]-1) == al_factor*(tie_ds.sizes['tie_rows']-1))

    # instrument data
    instrument_data = xr.open_dataset(dirname/'instrument_data.nc',
                                      mask_and_scale=False,
                                      # this variable has duplicate dimensions, drop it
                                      drop_variables='relative_spectral_covariance'
                                      ).chunk(chunks=chunks)
    for x in instrument_data.variables:
        ds[x] = instrument_data[x]

    # quality flags
    qf_file = dirname/'qualityFlags.nc'
    qf = xr.open_dataset(qf_file).chunk(chunks)
    for var in qf.variables: ds[var] = qf[var]

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
    meta = manifest['metadataSection']['metadataObject']
    date = meta[0]['metadataWrap']['xmlData']['acquisitionPeriod']
    start = datetime.fromisoformat(date['startTime'])
    stop  = datetime.fromisoformat(date['stopTime'])
    ds.attrs[n.datetime.name] = (start + (stop - start)/2.).isoformat()
    
    platform = meta[1]['metadataWrap']['xmlData']['platform']
    ds.attrs[n.platform.name] = platform['familyName'] + platform['number']
    ds.attrs[n.resolution.name] = 500
    ds.attrs[n.sensor.name] = platform['instrument']['familyName']['attributes']['abbreviation']
    ds.attrs[n.product_name.name] = dirname.name
    ds.attrs[n.input_directory.name] = str(dirname.parent)

    ds = ds.chunk(dict(detectors=-1))   # FIXME: do this upstream
    ds = ds.rename({'columns': n.columns.name, 'rows': n.rows.name})

    olci_init_spectral(ds, chunks)
    init_Rtoa(ds)

    if v1_compat: return _v1_compat(ds)
    return ds.unify_chunks()


def Level2_OLCI(dirname,
                chunks=500,
                tie_param=False,
                init_spectral=True,
                interp_angles='linear',
                ):
    '''
    Read an OLCI Level2 product as an xarray.Dataset
    '''
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
    ds = ds.assign({n.cwav.name: (('bands'),cwvl)})
    ds = ds.assign({n.bnames.name: (('bands'),bandnames)})
    
    # Retrieve product level
    text = manifest['informationPackageMap']['contentUnit']['attributes']
    levels = findall(r'Level .', text['textInfo'])
    assert len(levels) == 1, f'Invalid textinfo in manifest: "{text['textInfo']}"'
    level_from_manifest = levels[0].replace('Level ','level')
    assert 'level2' == level_from_manifest, \
        f'expected level2 encountered {level_from_manifest}'

    # Read main product
    ds = read_bands(ds, dirname, chunks, 2)

    # Geo coordinates
    geo_coords_file = dirname/'geo_coordinates.nc'
    geo = xr.open_dataset(geo_coords_file).chunk(chunks=chunks)
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
    tie_ds = xr.open_dataset(tie_geom_file).chunk(chunks=-1)
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
    tie = xr.open_dataset(tie_meteo_file).chunk(chunks=-1)
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
                                      mask_and_scale=False,
                                      # this variable has duplicate dimensions, drop it
                                      drop_variables='relative_spectral_covariance'
                                      ).chunk(chunks=chunks)
    for x in instrument_data.variables:
        ds[x] = instrument_data[x]

    # chl_nn
    fname = os.path.join(dirname, 'chl_nn.nc')
    qf = xr.open_dataset(fname).chunk(chunks=chunks)
    ds['chl_nn'] = qf.CHL_NN

    # chl_oc4me
    fname = os.path.join(dirname, 'chl_oc4me.nc')
    qf = xr.open_dataset(fname).chunk(chunks=chunks)
    ds['chl_oc4me'] = qf.CHL_OC4ME

    # quality flags
    fname = os.path.join(dirname, 'wqsf.nc')
    qf = xr.open_dataset(fname).chunk(chunks=chunks)
    ds['wqsf'] = qf.WQSF

    # aerosol properties
    fname = os.path.join(dirname, 'w_aer.nc')
    qf = xr.open_dataset(fname).chunk(chunks=chunks)
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
    ds.attrs[n.platform.name] = 'Sentinel-3'   # FIXME: A or B
    ds.attrs[n.sensor.name] = 'OLCI'
    ds.attrs[n.input_directory.name] = os.path.dirname(dirname)

    ds = ds.chunk(dict(detectors=-1))   # FIXME: do this upstream

    if init_spectral: olci_init_spectral(ds, chunks)

    ds = ds.rename({'columns': n.columns.name, 'rows': n.rows.name})

    return ds.unify_chunks()


def read_bands(ds: xr.Dataset, dirname: Path, chunks, level):
    
    prod_list = []
    for filename in dirname.glob('O*radiance.nc'):
        data = xr.open_dataarray(filename).chunk(chunks)
        prod_list.append(data)

    if level == 1: param_name = n.ltoa.name
    else: param_name = n.rho_w.name
    
    ds[param_name] = xr.concat(prod_list, dim=n.bands.name)
    return ds

def olci_init_spectral(ds, chunks):
    '''
    Broadcast all spectral (detector-wise) dataset to the whole image

    Adds the resulting datasets to `ds`: wav, F0 (in place)
    '''
    # wavelength
    ds[n.wav.name] = xr.apply_ufunc(
        lambda l0, di: l0[:,0,0,di],
        ds.lambda0,  # (bands x detectors)
        ds.detector_index,   # (rows x columns)
        dask='parallelized',
        input_core_dims=[['detectors'], []],
        output_dtypes=[ds.lambda0.dtype],
    )
    ds[n.wav.name].attrs.update(ds.lambda0.attrs)

    # solar flux
    ds[n.F0.name] = xr.apply_ufunc(
        lambda sf, di: sf[:,0,0,di],
        ds.solar_flux,  # (bands x detectors)
        ds.detector_index,   # (rows x columns)
        dask='parallelized',
        input_core_dims=[['detectors'], []],
        output_dtypes=[ds.solar_flux.dtype],
    )
    ds[n.F0.name].attrs.update(ds.solar_flux.attrs)


def decompose_flags(value, flags):
    '''
    return list of flag meanings for a given binary value
    flags: dictionary of meaning: value
    '''
    return [m for (m, v) in flags.items() if (v & value != 0)]


def get_valid_l2_pixels(wqsf, flags=[
        'INVALID', 'LAND', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN',
        'SNOW_ICE', 'SUSPECT', 'HISOLZEN', 'SATURATED', 'HIGHGLINT', 'WHITECAPS',
        'AC_FAIL', 'OC4ME_FAIL', 'ANNOT_TAU06', 'RWNEG_O2', 'RWNEG_O3', 'RWNEG_O4',
        'RWNEG_O5', 'RWNEG_O6', 'RWNEG_O7', 'RWNEG_O8', 'ANNOT_ABSO_D',
        'ANNOT_DROUT', 'ANNOT_MIXR1']):
    '''
    Get valid standard level2 pixels with a given flag set
    '''
    bval = 0
    L2_FLAGS = getflags(wqsf)
    for flag in flags:
        bval += int(L2_FLAGS[flag])

    return wqsf & bval == 0


def _v1_compat(ds):
    
    # Reset band coordinates
    ds = ds.assign_coords(bands=[400, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709, 754, 760, 764, 767, 779, 865, 885, 900, 940, 1020]) 
    
    # rename bands variable
    ds = ds.assign({n.rtoa.name: ((n.bands.name, n.rows.name, n.columns.name), ds[n.rtoa.name].data)})
    
    # Add flags
    ds[n.flags.name] = xr.zeros_like(
        ds.vza,
        dtype=n.flags.dtype)
    qf = getflags(ds.quality_flags)

    # raise LAND mask when land is raised but not fresh_inland_water
    from .eo import raiseflag
    raiseflag(
        ds[n.flags.name],
        "LAND", 1,
        ds.quality_flags & (qf["land"] + qf["fresh_inland_water"]) == qf["land"],
    )
    raiseflag(
        ds[n.flags.name],
        "L1_INVALID", 4,
        ds.quality_flags & qf["invalid"],
    )
    
    # Complete attributes
    for k,v in list(ds.longitude.attrs.items())[5:]: ds.attrs[k] = v
    
    return ds