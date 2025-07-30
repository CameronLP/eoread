#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

import dask.array as da
import pandas as pd
import xarray as xr

from eoread.eo import init_geometry as init_geo
from eoread.utils import spatial_resample, filter_metadata

from core.geo import n
from core.tools import raiseflag
from core.tools import merge, drop_unused_dims
from core import env, log


def get_sample(level: int=1, use_cache:bool=True) -> Path:
    """
    Bring a SGLI file path to test reading function

    Args:
        level (int, optional): Level of the product. Defaults to 1.
        use_cache (bool, optional): Option to save the result of the query to the download API to speed up the process. Defaults to True.
    """
    # Assumes that sample file exists locally in dir_samples
    # Downloaded from /standard/GCOM-C/GCOM-C.SGLI/L1B/2/2019/12/05
    sample = Path('/archive2/data/EOREAD_TESTDATA/SGLI/GC1SG1_201912050159F05712_1BSG_VNRDQ_1007.h5')
    assert sample.exists()
    return sample


sgli_central_wavelengths = da.array([
    380.00, 412.00, 443.00, 490.00,
    530.00, 565.00, 673.50, 673.50,
    763.00, 868.50, 868.50], dtype='float32')


def Level1_SGLI(filepath: str|Path,
                chunks: int|tuple = 500,
                metadata_template: list = None,
                add_ancillary_data: bool = False, 
                v1_compat: bool = False):
    """
    Read an SGLI Level1 product as an xarray.Dataset
    Formats the Dataset so that it contains the TOA radiances,
    the angles on the full grid, etc.

    Arguments:
        filepath: Path of the ECOSTRESS H5file (Ex: GC1SG1_201912050000N02307_1BSG_VNRDK_1007.h5)
        chunks: Size of chunks for spatial axis
        metadata_template: If None, add all metadata in output xarray.Dataset attributes else add only specified metadata.
        add_ancillary_data: Option to add ancillary data contained in provided file to the output dataset
        v1_compat: Option to format output xarray.Dataset such as version 1
    """
    
    ds = xr.Dataset()
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'

    # open image_data
    tree = xr.open_datatree(filepath, phony_dims='sort')
    imdata = tree['Image_data'].to_dataset()
    
    # read metadata
    log.debug('Reading metadata files')
    metadata = _read_metadata(ds, tree, metadata_template)
    ds = ds.assign({n.bnames.name: ((n.bands.name), metadata['Stored_channels'].split(',')),
                    n.cwav.name: ((n.bands.name), sgli_central_wavelengths)})

    imdata = imdata.rename_dims(dict(zip(
        imdata.Lt_VN01.dims,
        (n.rows.name, n.columns.name)
    )))
    shape = imdata.Lt_VN01.shape

    log.debug('Read and compute geometric angles')
    _init_geometry(ds, tree, shape, chunks)
    init_geo(ds)
    
    log.debug('Read top of atmosphere data')
    ds = _init_toa(ds, imdata, chunks)

    # Attributes
    log.debug('Add important attributes')
    ds.attrs[n.datetime.name] = metadata['Scene_center_time']
    ds.attrs[n.product_name.name] = metadata['Product_file_name']
    ds.attrs[n.platform.name] = 'GCOM-C'
    ds.attrs[n.sensor.name] = 'SGLI'
    ds.attrs[n.resolution.name] = 250
    ds.attrs[n.input_directory.name] = str(filepath.parent)
    
    # # Flags
    # ds[naming.flags] = xr.zeros_like(
    #     ds.vza,
    #     dtype=naming.flags_dtype)

    # raiseflag(
    #     ds[naming.flags],
    #     'LAND',
    #     flags['LAND'],
    #     imdata['Land_water_flag'] > thres_land_flag,
    # )
    
    if add_ancillary_data: ds = _read_ancillary(ds, tree)
    ds = ds.assign_coords({n.bands.name: ds[n.bands_nvis.name].data})

    return drop_unused_dims(ds).unify_chunks()


def _init_toa(ds, imdata, chunks):

    for i in range(len(ds.bands)):
        Rtoa = imdata[f'Lt_VN{i+1:02}'].chunk(chunks)
        attrs = Rtoa.attrs
        Rtoa = (Rtoa & attrs['Mask']) * attrs['Slope_reflectance'] + attrs['Offset_reflectance']
        Rtoa = Rtoa/ds.mus
        Rtoa.attrs = attrs
        ds[n.rtoa.name+f'_{i+1}'] = Rtoa
        
    ds = merge(ds, dim=n.bands_nvis.name)
    ds[n.rtoa.name].attrs['unit'] = None
    return ds


def _init_geometry(ds, tree, shape, chunks):
    
    geom = tree['Geometry_data'].to_dataset()

    geom = geom.rename_dims(dict(zip(
        geom.Latitude.dims,
        (n.rows.name+'_tie', n.columns.name+'_tie')
    )))

    ds['lat_tie'] = geom.Latitude
    ds['lon_tie'] = geom.Longitude

    ds['vza_tie'] = geom['Sensor_zenith']
    ds['vaa_tie'] = geom['Sensor_azimuth']
    ds['sza_tie'] = geom['Solar_zenith']
    ds['saa_tie'] = geom['Solar_azimuth']

    delta = 10
    for x in [x for x in ds if x.endswith('_tie')]:
        assert ds[x].Resampling_interval == delta
        assert ds[x].Offset == 0.
        ds[x] = ds[x] * ds[x].Slope

    # assign tiepoint coordinates
    ds[n.columns.name+'_tie'] = da.arange(ds.sizes[n.columns.name+'_tie'])*delta
    ds[n.rows.name+'_tie'] = da.arange(ds.sizes[n.rows.name+'_tie'])*delta

    # Create interpolated datasets
    shape = dict(zip(ds.lat_tie.dims, shape))
    for (name, A) in [
            (n.lat.name, ds.lat_tie),
            (n.lon.name, ds.lon_tie),
            (n.vza.name, ds.vza_tie),
            (n.vaa.name, ds.vaa_tie),
            (n.sza.name, ds.sza_tie),
            (n.saa.name, ds.saa_tie),
        ]:
        ds[name] = spatial_resample(A, shape, chunks=chunks)


def calc_central_wavelength():
    """
    Read SRF and calculate central wavelength for each band

    `print([f'{x:.2f}' for x in calc_central_wavelength()[1]])`

    Returns:
    --------
    sgli_bands: list of band identifiers

    wav_data: list of central wavelengths for each band
    """
    dir_auxdata = Path(__file__).parent/'auxdata'/'sgli'

    file_rsr = dir_auxdata/'sgli_rsr_f_for_algorithm_201008.txt.gz'
    assert file_rsr.exists(), file_rsr

    rsr = pd.read_csv(
        file_rsr,
        engine='python',
        delim_whitespace=True,
        index_col=False,
    )

    rsr = rsr.rename(columns=dict(zip(
        [x for x in rsr.columns if x.startswith('WL')],
        [x.replace('RSR_', 'WL_') for x in rsr.columns if x.startswith('RSR')],
    )))

    wav_data = []

    # calculate central wavelengths
    for i, _ in enumerate(sgli_bands):
        srf = rsr[f'RSR_VN{i+1:02}']
        wav = rsr[f'WL_VN{i+1:02}']
        wav_eq = da.trapz(wav*srf)/da.trapz(srf)
        wav_data.append(wav_eq)

    return sgli_bands, wav_data

def _read_metadata(ds, tree, template):
    filter_fn = (lambda x,y: x) if template is None else filter_metadata
    metadata = tree['Global_attributes'].attrs
    ds.attrs['metadata'] = filter_fn(metadata, template)
    return metadata

def _read_ancillary(ds, tree):
    log.info('Read ancillary data')
    ancillary = tree['Ancillary_data'].to_dict()
    for name, data in ancillary.items():
        for var, val in data.variables.items():
            n = '/'.join([name, var]) 
            ds = ds.assign({n:val})
    return ds