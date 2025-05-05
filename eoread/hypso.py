from pathlib import Path
import xarray as xr
# from eotools.solar_irradiance import solar_irradiance
from core.interpolate import interp, Linear
from dateutil.parser import parse
from eoread.utils.naming import naming
from core.naming import names


def Level1_HYPSO(level1_product: Path) -> xr.Dataset:
    """
    Open NTNU HYPSO Level1
    """
    ds = xr.Dataset()

    chunks = {"lines": 100, "samples": 100}
    ds_root = xr.open_dataset(level1_product)
    ds_products = xr.open_dataset(level1_product, group="products", chunks=chunks)
    ds_nav = xr.open_dataset(level1_product, group="navigation", chunks=chunks)

    # get _indirect geographical coordinates and angles if available
    ds["latitude"] = getattr(ds_nav, "latitude_indirect", ds_nav["latitude"])
    ds["longitude"] = getattr(ds_nav, "longitude_indirect", ds_nav["longitude"])
    ds["vza"] = getattr(ds_nav, "sensor_zenith_indirect", ds_nav["sensor_zenith"])
    ds["sza"] = getattr(ds_nav, "solar_zenith_indirect", ds_nav["solar_zenith"])
    ds["vaa"] = getattr(ds_nav, "sensor_azimuth_indirect", ds_nav["sensor_azimuth"])
    ds["saa"] = getattr(ds_nav, "solar_azimuth_indirect", ds_nav["solar_azimuth"])

    ds["Ltoa"] = xr.concat([ds_products[x] for x in ds_products], dim="bands")

    ds["wav"] = xr.DataArray(
        [ds_products[x].wavelength for x in ds_products], dims=["bands"]
    )
    ds = ds.assign_coords(bands=[int(ds_products[x].wave_name) for x in ds_products])

    # # read solar irradiance
    # F0 = solar_irradiance("LISIRD", variant="1nm")
    # ds["F0"] = interp(F0, wavelength=Linear(ds.wav))
    # # convert it to a unit compatible with Ltoa
    # assert ds.F0.units == "W m-2 nm-1"
    # ds["F0"] = ds.F0 * 1000
    # ds.F0.attrs.update(units="W m-2 um-1")
    # ds = ds.rename(lines="y", samples="x")

    # acquisition datetime
    ds.attrs.update(
        datetime=parse(ds_root.attrs["timestamp_acquired"], fuzzy=True).isoformat()
    )

    ds[naming.flags] = xr.zeros_like(ds.vza, dtype=naming.flags_dtype)
    ds.attrs.update(sensor=ds_root.sat_id)
    ds.attrs.update(product_name=level1_product.name)

    return ds
