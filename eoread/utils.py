from core.geo import n, convert_latlon
from core.interpolate import Linear, Nearest, interp
from core.files import uncompress
from dask.array import linspace
from tempfile import TemporaryDirectory
from typing import Literal
import xarray as xr


def filter_metadata(metadata: dict, template: list) -> dict:
    """
    Short method to filter metadata dictionary based on a template

    Args:
        metadata (dict): Dictionary corresponding to metadata
        template (list): List of list describing nodes to keep
    
    Examples:
        >> d = {'a': 0, 'b': {'c': 1, 'd':2}}
        >> t = [['a'], ['b','c']]
        >> filter_metadata(d, t)
        {'a': 0, 'b': {'c': 1}}
    """  
    
    # Populate new dictionary with nodes to kept 
    result = {}    
    for key_list in template:
        current_meta = metadata
        current_result = result
        
        # Traverse the nested dictionary using the provided keys
        for key in key_list[:-1]:
            if isinstance(current_meta, dict) and key in current_meta:
                current_result[key] = {}
                current_result = current_result[key]
                current_meta = current_meta[key]
            else: 
                raise
            
        # If the last key is present, add it to the result
        if isinstance(current_meta, dict) and key_list[-1] in current_meta:
            current_result.update({key_list[-1]: current_meta[key_list[-1]]})
    
    return result

def spatial_resample(arr,
                     output_shape: dict, 
                     chunks: int|list,
                     method= Literal['linear','repeat']):
    """
        ratio: list[x_ratio, y_ratio]
        if ratio is > 1 then downsamples
        if ratio is < 1 then upscales
    """
    arr = arr.squeeze()
    assert len(arr.shape) == 2, 'Array should be 2D'
    assert all(k in arr.dims for k in output_shape.keys())
        
    ratio = [arr.sizes[d]/output_shape[d] for d in arr.dims]
    
    if ratio[0] == 1 and ratio[1] == 1: 
        combi = zip(arr.dims, (n.rows.name, n.columns.name))
        return arr.rename({d0:d1 for d0,d1 in combi}) 
    
    # downsample
    if ratio[0] > 1. or ratio[1] > 1.:
        assert all(r == int(r) for r in ratio)
        params = {d: int(ratio[i]) for i,d in enumerate(arr.dims)}
        arr_resampled = arr.coarsen(**params, boundary="trim")
        arr_resampled = arr_resampled.mean().chunk(chunks)
        arr_resampled = arr_resampled.rename(**{b:a for b,a in zip(arr.dims, ['d0','d1'])})
        
    # over-sample
    else:
        m = Linear if method == 'linear' else Nearest
        
        raster = [linspace(0, arr.sizes[d], output_shape[d]) for d in arr.dims]
        raster = [xr.DataArray(l, dims=('d0','d1')).chunk(chunks) 
                  for l in convert_latlon(*raster)]
        params = {d: m(raster[i]) for i,d in enumerate(arr.dims)}
        arr_resampled = interp(arr.compute(), **params)

    return arr_resampled.rename({
        'd0': n.rows.name,
        'd1': n.columns.name})

def open_raster(dirname, 
                pattern: str, 
                compress_ext: str = None, 
                engine='h5netcdf'):
    """
    Methods to find a raster file based on a pattern and open it
    """
    # Find file path
    cloudfiles = list(dirname.glob(pattern))
    assert len(cloudfiles) == 1, \
        f'None or several files have been found for pattern "{pattern}"'
    path = cloudfiles[0]
    
    # Open and uncompress if needed 
    if compress_ext: 
        with TemporaryDirectory() as tmpdir:
            path = uncompress(path, tmpdir)
            return xr.open_dataarray(path, engine=engine).squeeze()
    return xr.open_dataarray(path, engine=engine).squeeze()