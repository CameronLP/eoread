#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Various utility functions for exploiting eoread objects
'''

import xarray as xr
import dask.array as da
from numpy import cos, radians, sqrt

# backward compatibility:
from core.files import to_netcdf
from core.tools import datetime
from core.tools import (contains, getflag, haversine, locate,
                        merge, raiseflag, split, sub, sub_pt, sub_rect,
                        wrap, getflags)
from core.geo import n


def init_Rtoa(ds: xr.Dataset):
    '''
    Initialize TOA reflectances from radiance (in place)

    Implies init_geometry
    '''
    init_geometry(ds)

    # TOA reflectance
    if n.rtoa.name not in ds:
        ds = ds.assign({n.rtoa.name: ((n.bands_nvis.name,n.rows.name,n.columns.name), 
                       (da.pi*ds[n.ltoa.name]/(ds.mus*ds[n.F0.name])).data)})
        ds[n.rtoa.name].attrs.update(unit=None)

    return ds

def scattering_angle(mu_s, mu_v, phi):
    """
    Scattering angle in degrees

    mu_s: cos of the sun zenith angle
    mu_v: cos of the view zenith angle
    phi: relative azimuth angle in degrees
    """
    sa = -mu_s*mu_v - sqrt((1.-mu_s*mu_s)*(1.-mu_v*mu_v)) * cos(radians(phi))
    return da.arccos(sa)*180./da.pi


def init_geometry(ds: xr.Dataset, 
                  scat_angle: bool =False):
    '''
    Initialize geometric variables (in place)
    '''

    # mus and muv
    if n.mus.name not in ds:
        ds[n.mus.name] = da.cos(da.radians(ds.sza))
        ds[n.mus.name].attrs['description'] = n.mus.desc
    if n.muv.name not in ds:
        ds[n.muv.name] = da.cos(da.radians(ds.vza))
        ds[n.muv.name].attrs['description'] = n.muv.desc

    # relative azimuth angle
    if n.raa.name not in ds:
        raa = ds[n.saa.name] - ds[n.vaa.name]
        raa = raa % 360
        ds[n.raa.name] = raa.where(raa < 180, 360-raa)
        ds.raa.attrs['description'] = n.raa.desc
        ds.raa.attrs['unit'] = n.raa.unit

    # scattering angle
    if scat_angle:
        ds['scat_angle'] = scattering_angle(ds.mus, ds.muv, ds.raa)
        ds['scat_angle'].attrs['description'] = 'scattering angle'

    return ds


def show_footprint(ds: xr.Dataset, 
                   zoom: int = 4):
    import ipyleaflet as ipy

    poly_pts = ds.attrs['Footprint']
    center = [sum(x)/len(poly_pts) for x in zip(*poly_pts)]

    m = ipy.Map(zoom=zoom,
                center=center)
    polygon = ipy.Polygon(locations=poly_pts,
                          color="green",
                          fillcolor="blue")
    m.add_layer(polygon)
    
    return m