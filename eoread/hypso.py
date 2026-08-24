from dask import array as da
from dateutil.parser import parse
from pathlib import Path
import numpy as np
import xarray as xr

from core.geo.naming import names
from core.interpolate import interp, Linear
from eoread.flags import FlagsReaderBase, GenericFlags
from eoread.tools import filter_metadata, format_chunks
from eotools.solar_irradiance import solar_irradiance
from core.tools import drop_unused_dims
from core import env, log


def Level1_HYPSO(
        filepath: str|Path,
        chunks: int|tuple = 500,
        metadata_template: list = None,
        v1_compat: bool = False,
        verbose: bool = True,
    ) -> xr.Dataset:
    """
    Read an NTNU HYPSO-1/HYPSO-2 Level1C product as an xarray.Dataset.

    HYPSO-1 and HYPSO-2 are hyperspectral imaging satellites operated by the
    Norwegian University of Science and Technology (NTNU). They provide
    high-resolution hyperspectral data for ocean monitoring and research.

    The dataset contains TOA radiances, viewing/solar angles on the full grid,
    and geolocation information.

    This reader targets the L1C NetCDF/HDF5 layout currently delivered by
    NTNU: a `products` group with one `Lt_<wavelength>` variable per band, a
    `geometry` group with the viewing/solar angles, and a
    `metadata/corrections` group carrying `radiometric_coefficients_version`
    ("original", "moved" or "adjusted"). That version is folded into the
    `sensor` attribute (e.g. "HYPSO-2_moved") so that `polymer.params.Params`
    can select the matching band set.

    Args:
        filepath: Path to the HYPSO L1C file (.nc)
        chunks: Size of chunks for spatial dimensions. If int, applies to both dimensions.
                If tuple, should be (rows_chunk, columns_chunk)
        metadata_template: List of metadata keys to include. If None, includes all metadata.
                          Use empty list [] for minimal metadata.
        v1_compat: If True, apply the eoread-v1 compatibility layer (adds a `flags` variable)
        verbose: If True, prints debug messages during reading

    Returns:
        xr.Dataset containing:
            - Ltoa: Top-of-atmosphere radiance (W/m^2/micrometer/sr)
            - F0: Extraterrestrial solar irradiance, interpolated onto the band wavelengths
            - vza, vaa, sza, saa: Viewing and solar geometry angles
            - latitude, longitude: Geolocation arrays
            - cwav: Central (nominal) wavelength per band
            - Metadata attributes, including `sensor` (e.g. "HYPSO-2_moved")

    Raises:
        AssertionError: If the file does not exist

    Example:
        >>> ds = Level1_HYPSO('hypso_product.nc', chunks=1000)
    """

    ds = xr.Dataset()
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'

    # Format chunks
    chunks = format_chunks(chunks)
    chunks_raw = {"lines": chunks[str(names.rows)], "samples": chunks[str(names.columns)]}

    ds_root = xr.open_dataset(filepath)
    ds_products = xr.open_dataset(filepath, group="products", chunks=chunks_raw)
    ds_nav = xr.open_dataset(filepath, group="geometry", chunks=chunks_raw)
    ds_corrections = xr.open_dataset(filepath, group="metadata/corrections")

    # get geographical coordinates and angles
    if verbose: log.debug('Read and compute geometric angles')
    ds[str(names.lat)] = ds_nav["latitude"]
    ds[str(names.lon)] = ds_nav["longitude"]
    ds[str(names.vza)] = ds_nav["sensor_zenith"]
    ds[str(names.sza)] = ds_nav["solar_zenith"]
    ds[str(names.vaa)] = ds_nav["sensor_azimuth"]
    ds[str(names.saa)] = ds_nav["solar_azimuth"]

    if verbose: log.debug('Read top of atmosphere data')
    ds[str(names.ltoa)] = xr.concat([ds_products[x] for x in ds_products], dim=str(names.bands))
    ds[str(names.ltoa)].attrs['units'] = 'W/m^2/micrometer/sr'
    ds = ds.rename(lines=str(names.rows), samples=str(names.columns))

    if verbose: log.debug('Extract central wavelength')
    ds = ds.assign_coords({
        str(names.bands): [int(ds_products[x].wave_name) for x in ds_products],
    })
    ds[str(names.cwav)] = xr.DataArray(
        [ds_products[x].wavelength for x in ds_products], dims=[str(names.bands)],
    )

    # HYPSO has no DEM/altitude data of its own; Polymer requires the
    # variable to be present (with pint-compatible units) when params.dem is
    # unset, hence the all-zero (sea-level) placeholder, dask-backed to match
    # the chunk grid of the other y/x variables.
    altitude = da.zeros((ds.sizes[str(names.rows)], ds.sizes[str(names.columns)]),
                         chunks=(ds.chunks[str(names.rows)], ds.chunks[str(names.columns)]))
    ds["altitude"] = ((str(names.rows), str(names.columns)), altitude, {"units": "m"})

    if verbose: log.debug('Compute solar irradiance')
    F0 = solar_irradiance("LISIRD", variant="1nm").compute()
    ds[str(names.F0)] = interp(F0, wavelength=Linear(ds[str(names.cwav)]))
    # convert it to a unit compatible with Ltoa
    assert ds[str(names.F0)].units == "W m-2 nm-1"
    ds[str(names.F0)] = ds[str(names.F0)] * 1000
    ds[str(names.F0)].attrs.update(units="W m-2 um-1")

    # Top of atmosphere reflectance. Computed here (rather than left to
    # eoread.eo.init_Rtoa) because that legacy helper discards the dataset it
    # rebuilds with `.assign` instead of mutating in place - every other
    # reader in this package works around it the same way, by providing Rtoa
    # directly.
    if verbose: log.debug('Compute top of atmosphere reflectance')
    mus = np.cos(np.radians(ds[str(names.sza)]))
    ds[str(names.rtoa)] = np.pi * ds[str(names.ltoa)] / (mus * ds[str(names.F0)])
    ds[str(names.rtoa)].attrs['unit'] = None

    # Add attributes
    if verbose: log.debug('Add important attributes')
    sensor_version = str(ds_corrections.radiometric_coefficients_version)
    ds.attrs[str(names.sensor)] = f"{ds_root.attrs['sat_id']}_{sensor_version}"
    ds.attrs[str(names.platform)] = 'HYPSO'
    ds.attrs[str(names.resolution)] = 40
    ds.attrs[str(names.product_name)] = filepath.name
    ds.attrs[str(names.input_directory)] = str(filepath.parent)
    ds.attrs[str(names.datetime)] = parse(ds_root.attrs['timestamp_acquired'], fuzzy=True).isoformat()
    ds.attrs['_flag_reader'] = 'eoread.hypso.FlagsReader_HYPSO'

    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs['metadata'] = filter_fn(ds_root.attrs, metadata_template)

    if v1_compat: return _v1_compat(ds)
    return drop_unused_dims(ds).unify_chunks()


class FlagsReader_HYPSO(FlagsReaderBase):
    """
    Placeholder flags reader for HYPSO L1C products.

    HYPSO L1C products carry no per-pixel L1 quality flags of their own (no
    land/water mask, no cloud mask, no invalid-pixel flag) - land and cloud
    masking are derived downstream by Polymer itself (GSW-based land mask,
    Rayleigh-based cloud mask). Every flag is therefore reported as
    unsupported, so `FlagsInit(strict=False)` leaves the initial flags at 0.
    """

    def requires(self) -> list[str]:
        return [str(names.vza)]

    def dims_like(self) -> str:
        return str(names.vza)

    def getflag(self, ds: xr.Dataset, flag_name: GenericFlags) -> xr.DataArray:
        raise ValueError(f'HYPSO L1C products do not provide a {flag_name} flag')


def Level1_HYPSO_future(
        filepath: str|Path,
        chunks: int|tuple = 500,
        metadata_template: list = None,
        v1_compat: bool = False,
        verbose: bool = True,
    ) -> xr.Dataset:
    """
    Read a HYPSO Level1 product as an xarray.Dataset, targeting a `navigation`/
    stacked-`Lt` product layout (a `navigation` group instead of `geometry`,
    and a single `Lt` variable stacked over a `bands` dimension instead of one
    `Lt_<wavelength>` variable per band).

    As of writing, NTNU's actual L1C deliveries use the older per-band
    `products`/`geometry` layout (see `Level1_HYPSO`), not this one, so this
    reader will fail to open them (`ds_root["navigation"]` / `ds_root["products"]["Lt"]`
    won't exist). It is kept here, unused, for whenever a delivery in this
    newer layout shows up.

    Args:
        filepath: Path to the HYPSO HDF5 file (.h5)
        chunks: Size of chunks for spatial dimensions. If int, applies to both dimensions.
                If tuple, should be (rows_chunk, columns_chunk)
        metadata_template: List of metadata keys to include. If None, includes all metadata.
                          Use empty list [] for minimal metadata.
        verbose: If True, prints debug messages during reading

    Returns:
        xr.Dataset containing:
            - Lt: Top-of-atmosphere radiance (W/sr/m^2)
            - VZA, VAA, SZA, SAA: Viewing and solar geometry angles
            - lat, lon: Geolocation arrays
            - central_wavelength: Band wavelengths
            - Metadata attributes

    Raises:
        AssertionError: If the file does not exist

    Example:
        >>> ds = Level1_HYPSO_future('hypso_product.h5', chunks=1000)
    """

    ds = xr.Dataset()
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'

    # Format chunks
    chunks = format_chunks(chunks)

    ds_root = xr.open_datatree(filepath, engine='h5netcdf')
    ds_products = ds_root["products"].to_dataset()
    ds_nav = ds_root["navigation"].to_dataset()

    # get _indirect geographical coordinates and angles if available
    if verbose: log.debug('Read and compute geometric angles')
    ds[str(names.lat)] = ds_nav["latitude"].chunk(chunks)
    ds[str(names.lon)] = ds_nav["longitude"].chunk(chunks)
    ds[str(names.vza)] = ds_nav["sensor_zenith"].chunk(chunks)
    ds[str(names.sza)] = ds_nav["solar_zenith"].chunk(chunks)
    ds[str(names.vaa)] = ds_nav["sensor_azimuth"].chunk(chunks)
    ds[str(names.saa)] = ds_nav["solar_azimuth"].chunk(chunks)

    if verbose: log.debug('Read top of atmosphere data')
    ds[str(names.ltoa)] = ds_products['Lt'].chunk(list(chunks)+[1])
    ds = ds.rename(lines=str(names.rows), samples=str(names.columns), bands=str(names.bands))
    ds[str(names.ltoa)].attrs['unit'] = 'W/sr/m^2'

    if verbose: log.debug('Extract central wavelength')
    ds = ds.assign_coords({
        str(names.bands): ds[str(names.bands)].data.astype(str),
    })
    ds = ds.assign({str(names.cwav): ((str(names.bands)), ds_products['Lt'].wavelengths)})

    # Add attributes
    if verbose: log.debug('Add important attributes')
    ds.attrs[str(names.sensor)] = ds_root.attrs['instrument']
    ds.attrs[str(names.platform)] = 'HYPSO'
    ds.attrs[str(names.resolution)] = 40
    ds.attrs[str(names.product_name)] = filepath.name
    ds.attrs[str(names.input_directory)] = str(filepath.parent)
    ds.attrs[str(names.datetime)] = ds_root.attrs['date_aquired']

    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs['metadata'] = filter_fn(ds_root.attrs, metadata_template)

    # ds[naming.flags] = xr.zeros_like(ds.vza, dtype=naming.flags_dtype)

    if v1_compat: return _v1_compat(ds)
    return drop_unused_dims(ds).unify_chunks()


def get_sample(level: int=1) -> Path:
    """
    Retrieve a sample HYPSO-1 product file for testing.

    Returns path to a pre-configured HYPSO sample product from environment variables.

    Args:
        level: Processing level of the product (currently only level=1 is supported)

    Returns:
        Path to the HYPSO HDF5 file

    Raises:
        AssertionError: If the sample directory does not exist

    Example:
        >>> hypso_file = get_sample(level=1)
        >>> ds = Level1_HYPSO(hypso_file)
    """
    sample = env.getdir('DIR_SAMPLE_HYPSO')
    assert sample.exists()
    return sample

def _v1_compat(ds):

    # Add flags
    ds["flags"] = xr.zeros_like(ds.vza, dtype="uint8")

    return ds
