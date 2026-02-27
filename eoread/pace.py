from pathlib import Path
from typing import Dict

import xarray as xr
from core.tools import only
from core.geo.naming import names
from dateutil.parser import parse

from eoread.tools import collect_sample, format_chunks, filter_metadata


def Level1B_PACE_OCI(product_pace_oci: Path) -> xr.Dataset:
    """
    Read PACE OCI Level 1B products
    """
    tree = xr.open_datatree(
        product_pace_oci,
        chunks={
            "scans": 300,
            "pixels": 200,
        },
    )

    geo = tree["geolocation_data"].to_dataset().reset_coords(["latitude", "longitude"])
    bdata = tree["sensor_band_parameters"].to_dataset()
    obs = tree["observation_data"].to_dataset()

    ds = xr.Dataset()
    ds["latitude"] = geo["latitude"]
    ds["longitude"] = geo["longitude"]

    ds["vaa"] = geo["sensor_azimuth"].astype("float32")
    ds["vza"] = geo["sensor_zenith"].astype("float32")
    ds["saa"] = geo["solar_azimuth"].astype("float32")
    ds["sza"] = geo["solar_zenith"].astype("float32")

    # TOA reflectance
    ds["Rtoa"] = xr.concat(
        [
            _Internal.rename_bands(obs["rhot_blue"]),
            _Internal.rename_bands(obs["rhot_red"]),
            _Internal.rename_bands(obs["rhot_SWIR"]),
        ],
        dim="bands",
    )

    ds["wav"] = xr.concat(
        [
            _Internal.rename_bands(bdata["blue_wavelength"]),
            _Internal.rename_bands(bdata["red_wavelength"]),
            _Internal.rename_bands(bdata["SWIR_wavelength"]),
        ],
        dim="bands",
    )

    # Flags
    ds["flags"] = xr.zeros_like(ds.sza, dtype="uint16")
    raiseflag(
        ds["flags"],
        "LAND",
        1,
        geo["watermask"] == 0,
    )

    # Attributes
    ds.attrs.update(sensor="OCI")
    ds.attrs.update(product_name=product_pace_oci.name)
    time_start = parse(tree.attrs["time_coverage_start"])
    time_end = parse(tree.attrs["time_coverage_end"])
    ds.attrs[str(names.datetime)] = (time_start + (time_end - time_start) / 2).isoformat()
    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    ds.attrs['metadata'] = filter_fn(tree.attrs, metadata_template)
    ds.attrs['_flag_reader'] = 'eoread.pace.FlagsReader_PACE'

    # # SRF getter
    # ds.attrs['_srf_getter'] = 'None'
    # ds.attrs['_srf_getter_arg'] = ''
    
    # Add band names as coordinates
    ds = ds.assign_coords({str(names.bands): ds[str(names.cwav)].astype(int).astype(str)})
    ds = ds.chunk({str(names.bands): -1})

    # x/y dimensions
    ds = ds.rename(scans="y", pixels="x")

    # Remove duplicate bands by keeping only unique wavelength values, sorted by increasing wavelength
    seen = set()
    ds = ds.isel(bands=[
        i for i, w in sorted([
            (i, w) for i, w in enumerate(ds.bands.values)
            if w not in seen and not seen.add(w)
        ], key=lambda x: x[1])
    ])

    return ds


def get_sample(sample: int = 1) -> Dict:
    """
    Return sample PACE Level-1B products

    Returns: a dict with keys:
        path: path to the product
        roi: region of interest within the full product
        px: sample pixel coordinates within the roi
    """

################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def rename_bands(ds: xr.Dataset) -> xr.Dataset:
        """Rename band dimension and add band group coordinate."""
        
        # Determine band dimension
        dim = only([d for d in ds.dims if 'bands' in d])
        
        # Rename bands dimension
        ds = ds.rename({dim: str(names.bands)})
        
        # Assign bands group dimension
        bgroup = [dim] * len(ds[str(names.bands)])
        ds = ds.assign_coords({str(names.bgroup): (str(names.bands), bgroup)})
        
        return ds