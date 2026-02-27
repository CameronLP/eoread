from core.interpolate import interp, Linear, Nearest
from core.import_utils import import_module
from core.uncompress import uncompress
from core.geo.naming import names
from core.tools import only
from core import env

from typing import Union, Literal
from xarray import DataArray, open_dataarray
from dask.array import meshgrid, linspace
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np


def format_chunks(chunks: Union[int, list, tuple, dict]) -> dict:
    """
    Normalize chunk specification to a dictionary format.
    
    Converts various chunk size specifications (int, list, tuple) to a
    standardized dictionary with 'rows' and 'columns' keys.
    
    Args:
        chunks: Chunk size specification:
                - int: Same chunk size for both dimensions
                - list/tuple: [rows_chunk, columns_chunk]
                - dict: Already in correct format
                
    Returns:
        Dictionary with 'rows' and 'columns' chunk sizes
        
    Raises:
        AssertionError: If chunks format is invalid
        
    Example:
        >>> format_chunks(500)
        {'rows': 500, 'columns': 500}
        >>> format_chunks([1000, 500])
        {'rows': 1000, 'columns': 500}
    """
    
    # Manage different chunk types
    if isinstance(chunks, int):
        chunks = [chunks, chunks]
    if isinstance(chunks, list|tuple):
        assert len(chunks) == 2
        chunks = {str(names.rows): chunks[0], str(names.columns): chunks[1]}
    
    assert len(chunks) == 2, 'chunks should be for spatial dimensions'
    assert str(names.rows) in chunks and str(names.columns) in chunks
    return chunks


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
                raise ValueError(f'{key} not found')
            
        # If the last key is present, add it to the result
        if isinstance(current_meta, dict) and key_list[-1] in current_meta:
            current_result.update({key_list[-1]: current_meta[key_list[-1]]})
        else:
            raise ValueError(f'Leaf {key_list[-1]} has been found')
    
    return result


def spatial_resample(
        array: DataArray,
        output_shape: dict, 
        chunks: dict,
        method: Literal['linear','repeat'] = 'linear'
    ) -> DataArray:
    """
        ratio: list[x_ratio, y_ratio]
        if ratio is > 1 then downsamples
        if ratio is < 1 then upscales
    """
    
    # Check inputs compliance
    array = array.squeeze()
    assert len(array.shape) == 2, 'Array should be 2D'
    assert all(k in array.dims for k in output_shape.keys())
    assert chunks.keys() == output_shape.keys()
    
    # Compute the ratio of transformation
    dims = array.dims
    ratio = {d: array.sizes[d]/output_shape[d] for d in dims}
    
    # No resolution change => return the input DataArray
    if all(v == 1 for v in ratio.values()):
        return array
    
    # downsample
    if any(v > 1 for v in ratio.values()):
        assert all(r == int(r) for r in ratio.values())
        params = {d: int(ratio[d]) for d in dims}
        resampled = array.coarsen(**params, boundary="trim")
        resampled = resampled.mean().chunk(chunks)
        
    # over-sample
    else:
        coords = array.coords
        method = Linear if method == 'linear' else Nearest
        xy = meshgrid(*[
            linspace(coords[d].values[0], coords[d].values[-1], output_shape[d]) 
            for d in dims
        ])
        params = {d: method(DataArray(xy[i], dims=('d1','d0'))) for i,d in enumerate(dims)}
        resampled = interp(array, **params)
        resampled = resampled.rename(d0=dims[0], d1=dims[1])
        
        # new = {d: np.linspace(
        #     coords[d].values[0], coords[d].values[-1], output_shape[d]
        # ) for d in dims}
        # params = {d: method(DataArray(new[d], dims=(d))) for d in dims}
        # resampled = interp(array, **params)
    
    return resampled


def open_raster(
        dirname: Union[str, Path], 
        pattern: str, 
        compress_ext: str = None, 
        engine='h5netcdf'
    ) -> DataArray:
    """
    Find and open a raster file matching a pattern.
    
    Searches for a single file matching the pattern and optionally
    decompresses it before opening.
    
    Args:
        dirname: Directory to search in
        pattern: Glob pattern to match (e.g., '*_B01.jp2')
        compress_ext: If provided, uncompresses file with this extension first
        engine: xarray engine for opening the file (default: 'h5netcdf')
        
    Returns:
        DataArray with the raster data (squeezed to remove size-1 dimensions)
        
    Raises:
        AssertionError: If pattern matches zero or multiple files
        
    Example:
        >>> arr = open_raster('/data/', '*_B02.tif', engine='rasterio')
        >>> arr = open_raster('/data/', '*_CLD.zip', compress_ext='.zip')
    """
    # Find file path
    path = only(list(Path(dirname).glob(pattern)))
    
    # Open and uncompress if needed 
    if compress_ext: 
        with TemporaryDirectory() as tmpdir:
            path = uncompress(path, tmpdir)
            return open_dataarray(path, engine=engine).squeeze()
    
    return open_dataarray(path, engine=engine).squeeze()


def collect_sample(
        variable: str,
        provider: Literal[None, 'cnes'],
        sand_collection: str|None = None,
        level: int|None = None
    ) -> Path:
    
    # Check if user has provided a path
    variable = env.getvar(variable, default='')  
    
    # If not provided, try to download a sample with SAND
    if variable == '':
        
        if provider is not None:
            assert sand_collection and level, 'Provide kwargs for SAND downloader'
        else: 
            raise ValueError('')
        
        # Check SAND importation
        try: 
            from sand.sample_product import products
        except ImportError:
            raise ImportError('To use get_sample function, you need to install SAND module')
        
        # Collect appropriated provider
        downloader = {
            'cdse': 'sand.copernicus_dataspace.DownloadCDSE',
            'cnes': 'sand.cnes.DownloadCNES',
            'eumdac': 'sand.eumdac.DownloadEumDAC',
            'nasa': 'sand.nasa.DownloadNASA',
            'usgs': 'sand.usgs.DownloadUSGS',
        }[provider]
        
        # Retrieve name of example product
        prod_id = products[sand_collection][f'l{level}_product']
        
        # Download product with SAND
        dl = import_module(downloader)()
        directory = env.getdir('DIR_SAMPLES')/sand_collection
        target = dl.download_file(prod_id, directory)
        
        assert target.exists()
        return target
        
    else:
        return Path(variable)
    
    
# FIXME : Should be move to core.tests.plots

from core.interpolate import Linear, Nearest, interp
from core.files import uncompress
from tempfile import TemporaryDirectory
from typing import Literal
import xarray as xr
import numpy as np


def xrimshow(
    da: xr.DataArray,
    title: str | None=None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    display_size: float=3.0,
    margin_top: float=0.1,
    margin_left: float=0.6,
    margin_bottom: float=0.5,
    margin_right: float=0.1,
    cbar_thickness: float=0.15,
    cbar_length: float=0.5,
    cbar_gap_horizontal: float=0.5,
    cbar_gap_vertical: float=0.2,
    cbar_label_space: float=0.5,
    title_height: float=0.4,
    yincrease: bool = True,
):
    """
    Plot a 2D DataArray using imshow with consistent absolute margins.

    The figure size adapts to the data aspect ratio while maintaining
    constant image area (width * height = display_size²).  A square
    image is therefore display_size × display_size.

    The colorbar is placed below the image for wide data (aspect >= 1)
    and to the right for tall data (aspect < 1).  All margins, gaps and
    the colorbar thickness are specified in inches and stay constant
    regardless of data shape, ensuring a visually consistent layout.

    Parameters
    ----------
    da : xarray.DataArray
        2D DataArray to plot.
    title : str, optional
        Title for the plot.
    vmin, vmax : float, optional
        Colormap range.
    cmap : str, optional
        Colormap name (e.g., 'viridis', 'plasma'). If None, uses xarray's default colormap.
    display_size : float, default=3.
        Controls display size.  The image area in inches² equals
        display_size², so a square image would be
        display_size × display_size.
    margin_top : float, default=0.1
        Top margin in inches (used when there is no title).
    margin_left : float, default=0.5
        Left margin in inches.
    margin_bottom : float, default=0.3
        Bottom margin in inches.
    margin_right : float, default=0.1
        Right margin in inches.
    cbar_thickness : float, default=0.15
        Colorbar thickness in inches.
    cbar_length : float, default=0.5
        Colorbar length as a fraction of the image extent along the
        same axis (0 to 1).  The colorbar is centered.
    cbar_gap_horizontal : float, default=0.5
        Gap between the image and a horizontal colorbar (below) in inches.
    cbar_gap_vertical : float, default=0.2
        Gap between the image and a vertical colorbar (right) in inches.
    cbar_label_space : float, default=0.4
        Extra space in inches reserved for colorbar tick labels.
        Added to the right for vertical colorbars, to the bottom for
        horizontal colorbars.
    title_height : float, default=0.4
        Space reserved for the title in inches (replaces margin_top
        when *title* is set).
    yincrease : bool, default=True
        If True, the y-axis increases upward (standard orientation).
        If False, the y-axis increases downward (inverted orientation).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    im : matplotlib.image.AxesImage
    """
    from matplotlib import pyplot as plt

    # Data aspect ratio (width / height in pixels)
    aspect = da.shape[1] / da.shape[0]

    # Image dimensions in inches, preserving area = display_size²
    img_w = (display_size ** 2 * aspect) ** 0.5
    img_h = (display_size ** 2 / aspect) ** 0.5

    if title is None and isinstance(da.name, str):
        title = da.name

    top_space = title_height if title else margin_top

    # Set vmin/vmax
    da = da.compute()
    vmin = vmin or float(np.nanpercentile(da, 5))
    vmax = vmax or float(np.nanpercentile(da, 95))

    # Compute figure size and axes rectangles [left, bottom, w, h]
    if aspect >= 1:
        # Wide image → horizontal colorbar below
        fig_w = margin_left + img_w + margin_right
        fig_h = top_space + img_h + cbar_gap_horizontal + cbar_thickness + cbar_label_space

        img_rect = (
            margin_left / fig_w,
            (cbar_label_space + cbar_thickness + cbar_gap_horizontal) / fig_h,
            img_w / fig_w,
            img_h / fig_h,
        )
        cbar_actual_w = img_w * cbar_length
        cbar_x = margin_left + (img_w - cbar_actual_w) / 2
        cbar_rect = (
            cbar_x / fig_w,
            cbar_label_space / fig_h,
            cbar_actual_w / fig_w,
            cbar_thickness / fig_h,
        )
        cbar_orientation = 'horizontal'
    else:
        # Tall image → vertical colorbar on the right
        fig_w = margin_left + img_w + cbar_gap_vertical + cbar_thickness + cbar_label_space + margin_right
        fig_h = top_space + img_h + margin_bottom

        img_rect = (
            margin_left / fig_w,
            margin_bottom / fig_h,
            img_w / fig_w,
            img_h / fig_h,
        )
        cbar_actual_h = img_h * cbar_length
        cbar_y = margin_bottom + (img_h - cbar_actual_h) / 2
        cbar_rect = (
            (margin_left + img_w + cbar_gap_vertical) / fig_w,
            cbar_y / fig_h,
            cbar_thickness / fig_w,
            cbar_actual_h / fig_h,
        )
        cbar_orientation = 'vertical'

    # Create figure with dedicated image and colorbar axes
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(img_rect)
    cax = fig.add_axes(cbar_rect)

    # Plot using xarray's imshow
    im = da.plot.imshow(
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        add_colorbar=True,
        cbar_ax=cax,
        cbar_kwargs={'orientation': cbar_orientation},
        yincrease=yincrease,
    )

    if title is not None:
        ax.set_title(title, pad=3)

    return fig, ax, im


def downsample(da: xr.DataArray, size: int = 500) -> xr.DataArray:
    """
    Downsample a DataArray by striding so that its smallest
    dimension has between `size` and `2 x size` elements.

    Assigns arange coordinates to any dimension that lacks them,
    so that the strided result preserves meaningful axis values.

    Parameters
    ----------
    da : xr.DataArray
        Input array of any number of dimensions.
    size : int, default=500
        Target approximate number of elements along the smallest dimension.

    Returns
    -------
    xr.DataArray
        Strided view of the input with reduced resolution.
    """
    # Add arange coordinates for dimensions that have none
    for dim, dimsize in zip(da.dims, da.shape):
        if dim not in da.coords:
            da = da.assign_coords({dim: np.arange(dimsize)})
    m = min(da.shape)
    stride = max(1, m//size)
    s = slice(None, None, stride)
    return da[(s,)*da.ndim]