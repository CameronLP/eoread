from dask import array as da
from pathlib import Path
import xarray as xr

from eoread.utils import filter_metadata
# from eotools.solar_irradiance import solar_irradiance
from core.geo import n
from core.tools import  drop_unused_dims
from core import env, log


    ds = xr.Dataset()
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'

    ds_root = xr.open_datatree(filepath)
    if isinstance(chunks, int): chunks = [chunks]*2
    ds_products = ds_root["products"].to_dataset()
    ds_nav = ds_root["navigation"].to_dataset()

    # get _indirect geographical coordinates and angles if available
    log.debug('Read and compute geometric angles')
    ds[n.lat.name] = ds_nav["latitude"].chunk(chunks)
    ds[n.lon.name] = ds_nav["longitude"].chunk(chunks)
    ds[n.vza.name] = ds_nav["sensor_zenith"].chunk(chunks)
    ds[n.sza.name] = ds_nav["solar_zenith"].chunk(chunks)
    ds[n.vaa.name] = ds_nav["sensor_azimuth"].chunk(chunks)
    ds[n.saa.name] = ds_nav["solar_azimuth"].chunk(chunks)

    log.debug('Read top of atmosphere data')
    ds[n.ltoa.name] = ds_products['Lt'].chunk(list(chunks)+[1])
    ds = ds.rename(lines=n.rows.name, samples=n.columns.name, bands=n.bands.name)
    ds[n.ltoa.name].attrs['unit'] = 'W/sr/m^2'
    
    log.debug('Extract central wavelength')
    ds = ds.assign_coords({n.bands.name: da.arange(len(ds[n.bands.name]))+1})
    ds = ds.assign({n.cwav.name: ((n.bands.name), ds_products['Lt'].wavelengths),
         n.bnames.name: ((n.bands.name), ds[n.bands.name].data.astype(str))})

    # # read solar irradiance
    # F0 = solar_irradiance("LISIRD", variant="1nm")
    # ds["F0"] = interp(F0, wavelength=Linear(ds.wav))
    # # convert it to a unit compatible with Ltoa
    # assert ds.F0.units == "W m-2 nm-1"
    # ds["F0"] = ds.F0 * 1000
    # ds.F0.attrs.update(units="W m-2 um-1")
    # ds = ds.rename(lines="y", samples="x")

    # acquisition datetime
    log.debug('Add important attributes')
    ds.attrs[n.sensor.name] = ds_root.attrs['instrument']
    ds.attrs[n.platform.name] = 'HYPSO'
    ds.attrs[n.resolution.name] = 40
    ds.attrs[n.product_name.name] = filepath.name
    ds.attrs[n.input_directory.name] = str(filepath.parent)
    ds.attrs[n.datetime.name] = ds_root.attrs['date_aquired']
    
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs['metadata'] = filter_fn(ds_root.attrs, metadata_template)

    # ds[naming.flags] = xr.zeros_like(ds.vza, dtype=naming.flags_dtype)
    
    return drop_unused_dims(ds).unify_chunks()


def get_sample(level: int=1, use_cache:bool=True) -> Path:
    sample = Path('/mnt/ceph/user/francois/HYPSO/sample1/vancouver_2022-07-30_1825Z-l1b.nc')
    assert sample.exists()
    return sample