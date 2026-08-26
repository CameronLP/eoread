from dask import array as da
from dateutil.parser import parse
from pathlib import Path
import h5py
import numpy as np
import xarray as xr

from core.geo.naming import names
from core.interpolate import interp, Linear
from eoread.flags import FlagsReaderBase, GenericFlags
from eoread.tools import filter_metadata, format_chunks
from eotools.solar_irradiance import solar_irradiance
from core.tools import drop_unused_dims
from core import env, log


def _hypso_format(filepath: Path) -> str:
    """"grouped" (NTNU's original delivery layout: separate `products`/
    `geometry` HDF5 groups) or "flat" (hypso-package's CF/SNAP-refactored
    layout, in production since 2026-08: products+geometry variables at the
    file root, only `metadata/*` stays nested - see that package's
    REFACTOR_PROGRESS.md/ARCHITECTURE_PROPOSAL.md for why). A cheap,
    dedicated open+close - no data read - since the two layouts need
    different xr.open_dataset(..., group=...) calls below, not something
    detectable from one already-open Dataset."""
    with h5py.File(filepath, "r") as f:
        return "grouped" if "products" in f else "flat"


def _read_hypso_l1c_products_and_geometry(filepath: Path, chunks_raw: dict, fmt: str):
    """The one part of reading a HYPSO L1C file that actually differs between
    layouts (see _hypso_format) - everything downstream in Level1_HYPSO
    (F0/Rtoa computation, attribute assembly) is identical either way, so
    only this extraction step is duplicated, not the science.

    Returns (ds_root, ds_nav, ltoa, wave_names, wavelengths):
      - ds_root: the whole file's global attrs (unchunked - attrs don't care)
      - ds_nav: Dataset exposing latitude/longitude/sensor_zenith/etc by name
        (a dedicated "geometry" group for "grouped"; ds_root itself for
        "flat", since geometry variables already live at its root)
      - ltoa: (bands, lines, samples) DataArray, band dim ordered by each
        band variable's own `band` attribute (not name/insertion order -
        the confirmed latent band-order bug this convention now guards
        against, see hypso-package's tests/test_cf_format.py)
      - wave_names, wavelengths: per-band label/value lists, same order as
        ltoa's band dimension
    """
    if fmt == "grouped":
        ds_root = xr.open_dataset(filepath)
        ds_products = xr.open_dataset(filepath, group="products", chunks=chunks_raw)
        ds_nav = xr.open_dataset(filepath, group="geometry", chunks=chunks_raw)
    else:
        ds_root = xr.open_dataset(filepath, chunks=chunks_raw)
        # latitude/longitude get auto-promoted to coordinate variables by
        # xarray's CF decoding (every Lt_<wave> variable carries a
        # coordinates="latitude longitude" attribute pointing at them, now
        # resolvable since they live in the same root group - see
        # hypso-package's io/cf.py geolocation_ref_attrs()). Demoted back to
        # plain data variables so ds_nav["longitude"] below assigns into the
        # caller's fresh Dataset the same way the "grouped" branch's
        # never-promoted ds_nav["longitude"] does - otherwise xarray can't
        # tell whether the assignment target should be a coord or not and
        # raises MergeError.
        ds_root = ds_root.reset_coords(["latitude", "longitude"])
        ds_products = ds_root
        ds_nav = ds_root

    if "Lt" in ds_products:
        # Single stacked (lines, samples, bands) datacube variable - only
        # possible with "flat" (hypso-package's write_level_nc(datacube=True)
        # option; "grouped" deliveries observed in practice are always
        # per-band, see below), but checked unconditionally since nothing
        # about the file format guarantees which one a given delivery used.
        ltoa = ds_products["Lt"].transpose(str(names.bands), ...)
        wavelengths = [float(w) for w in ltoa.attrs["wavelengths"]]
        wave_names = [str(int(round(w))) for w in wavelengths]
    else:
        band_vars = sorted(
            (v for v in ds_products.data_vars if v.startswith("Lt_")),
            key=lambda v: int(ds_products[v].attrs["band"]),
        )
        ltoa = xr.concat([ds_products[v] for v in band_vars], dim=str(names.bands))
        wave_names = [str(ds_products[v].wave_name) for v in band_vars]
        wavelengths = [float(ds_products[v].wavelength) for v in band_vars]

    return ds_root, ds_nav, ltoa, wave_names, wavelengths


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

    Transparently supports both HYPSO L1C NetCDF/HDF5 layouts NTNU has
    delivered (auto-detected per file, see _hypso_format): the original
    "grouped" layout (a `products` group with one `Lt_<wavelength>` variable
    per band, a `geometry` group with the viewing/solar angles) and the
    current "flat" layout (hypso-package's CF/SNAP refactor, in production
    since 2026-08: products+geometry variables at the file root instead of
    nested groups). Either way, a `metadata/corrections` group (unchanged by
    the format switch) carries `radiometric_coefficients_version` ("original",
    "moved" or "adjusted"), folded into the `sensor` attribute (e.g.
    "HYPSO-2_moved") so that `polymer.params.Params` can select the matching
    band set. Also supports a file written with a single stacked
    (lines, samples, bands) `Lt` datacube variable instead of one `Lt_<wave>`
    per band (only possible in the flat layout - see
    _read_hypso_l1c_products_and_geometry).

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

    fmt = _hypso_format(filepath)
    if verbose: log.debug('Detected HYPSO L1C layout: %s', fmt)
    ds_root, ds_nav, ltoa, wave_names, wavelengths = _read_hypso_l1c_products_and_geometry(
        filepath, chunks_raw, fmt)
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
    ds[str(names.ltoa)] = ltoa
    ds[str(names.ltoa)].attrs['units'] = 'W/m^2/micrometer/sr'
    ds = ds.rename(lines=str(names.rows), samples=str(names.columns))

    if verbose: log.debug('Extract central wavelength')
    ds = ds.assign_coords({
        str(names.bands): [int(w) for w in wave_names],
    })
    ds[str(names.cwav)] = xr.DataArray(wavelengths, dims=[str(names.bands)])

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
