from eoread.utils import filter_metadata, spatial_resample
from eoread.hdf4 import load_hdf4
from typing import Union, Literal
from pathlib import Path

from core.geo import n
from core import env, log
from core.tools import drop_unused_dims, merge

import xarray as xr
import dask.array as da



user_guide = 'https://mcst.gsfc.nasa.gov/sites/default/files/file_attachments/M1054E_PUG_2022_1005_V6.2.2_Terra_V6.2.3_Aqua.pdf'

bnames = [1,2,3,4,5,6,7,8,9,10,11,12,13,13.5,14,14.5,15,16,17,18,19,20,21,22,
          23,24,25,26,27,28,29,30,31,32,33,34,35,36]

cwvl = [650, 860, 470, 555, 1240, 1640, 2130, 410, 440, 485, 530, 550, 670, 672, 
        680, 682, 750, 870, 900, 935, 940, 3750, 3960, 3962, 4050, 4460, 4510, 
        1375, 6710, 7230, 8550, 9730, 11000, 12000, 13230, 13630, 13930, 14230]

# Planck's law constants 
h = 6.6260755e-34
c = 2.9979246e+8 # (meters per second)
k = 1.380658e-23 # (Joules per Kelvin)

# derived constants
K1 = 2.0 * h * c * c
K2 = h * c / k


def Level1_MODIS(
        filepath: Union[Path, str], 
        chunks: int = 100,
        resolution: Literal[250,500,1000,None] = 250,
        metadata_template: Union[list, None] = None,
        v1_compat: bool = False,
        verbose: bool = True
    ) -> xr.Dataset:
    """
    Read a MODIS Level1B product as an xarray.Dataset.
    
    MODIS (Moderate Resolution Imaging Spectroradiometer) provides 36 spectral
    bands from visible to thermal infrared with spatial resolutions of 250m,
    500m, and 1000m.

    Args:
        filepath: Path to the MODIS HDF4 file (.hdf)
        chunks: Size of chunks for spatial dimensions
        resolution: Resample all bands to provided resolution and concatenate.
                If None, keep original resolutions (250m, 500m, 1km) separate.
        metadata_template: List of metadata keys to include. If None, includes all metadata.
                          Use empty list [] for minimal metadata.
        v1_compat: If True, formats output to match version 1 structure
        
    Example:
        >>> ds = Level1_MODIS('MOD021KM.A2023001.0000.061.*.hdf')
    """
    
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'
    if isinstance(chunks, int): chunks = [chunks]*2   
    
    # Revize variables
    if verbose: log.debug('Reading h4file')
    l1 = load_hdf4(filepath, trim_dims=True)
    l1 = l1.rename_vars({
        'Latitude': str(n.lat), 'Longitude': str(n.lon),
        'SensorZenith': str(n.vza), 'SensorAzimuth': str(n.vaa),
        'SolarZenith': str(n.sza), 'SolarAzimuth': str(n.saa)
    })
 
    # Read metadata
    metadata = {}
    if verbose: log.debug('parsing metadata text')
    for name in ['CoreMetadata.0','ArchiveMetadata.0','StructMetadata.0']:
        p = _parser_attrs(l1.attrs[name].split('\n'))
        p.parse()
        metadata.update(p.data)
    
    # Add band information
    if verbose: log.debug('Add central wavelength')
    l1 = l1.assign_coords({str(n.bands): da.array(bnames).astype(str)})
    l1 = l1.assign({str(n.cwav): ((str(n.bands)), cwvl)})
    
    # Rescale angles data
    if verbose: log.debug('Read and compute geometric angles')
    for varname in [str(n.vza), str(n.vaa), str(n.sza), str(n.saa)]:
        l1[varname] = l1[varname].scale_factor * l1[varname]

    # Change radiometry of input data   
    if verbose: log.debug('Read top of atmosphere data')
    l1 = _transform_radiometry(l1, chunks, resolution)
    l1 = _aggregate_vars(l1, chunks, resolution)
    l1 = _compute_bt(l1, resolution)
    
    # Upscale latlon variables
    if verbose: log.debug('upscale latlon variables')
    shape = {k:v for k,v in zip(l1[str(n.lat)].dims, l1[str(n.bt)].shape[1:])}
    l1[str(n.lat)] = spatial_resample(l1[str(n.lat)], shape, chunks)
    l1[str(n.lon)] = spatial_resample(l1[str(n.lon)], shape, chunks)

    # Change dimensions name and update coordinates
    l1 = _rename_dims(l1)

    # Summarize Attributes
    if verbose: log.debug('Add important attributes')
    attributes = l1.attrs
    
    l1.attrs = {}
    l1.attrs[str(n.input_directory)] = str(filepath.parent)
    l1.attrs[str(n.resolution)]   = resolution
    l1.attrs[str(n.datetime)]     = metadata['INVENTORYMETADATA']['ECSDATAGRANULE']['PRODUCTIONDATETIME']['VALUE'][1:-1]
    l1.attrs['night']             = str(metadata['INVENTORYMETADATA']['ECSDATAGRANULE']['DAYNIGHTFLAG']['VALUE'] != '"Day"')
    l1.attrs[str(n.product_name)] = metadata['INVENTORYMETADATA']['ECSDATAGRANULE']['LOCALGRANULEID']['VALUE'][1:-1]
    l1.attrs[str(n.platform)]     = metadata['INVENTORYMETADATA']['ASSOCIATEDPLATFORMINSTRUMENTSENSOR']['ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']['ASSOCIATEDPLATFORMSHORTNAME']['VALUE'][1:-1]
    l1.attrs[str(n.sensor)]       = metadata['INVENTORYMETADATA']['ASSOCIATEDPLATFORMINSTRUMENTSENSOR']['ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']['ASSOCIATEDSENSORSHORTNAME']['VALUE'][1:-1]
    l1.attrs[str(n.shortname)]    = metadata['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE'][1:-1]
    l1.attrs['version']           = int(metadata['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['VERSIONID']['VALUE'])
    l1.attrs['user_guide']        = user_guide

    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    metadata['attributes'] = attributes
    l1.attrs['metadata'] = filter_fn(metadata, metadata_template)
    
    l1 = drop_unused_dims(l1)
    band_dims = [b for b in l1.dims if b.startswith(str(n.bands)+'_')]
    bands = [list(l1[b].values) for b in band_dims]
    groups = [[b]*l1.sizes[b] for b in band_dims]
    indexes = [list(l1[str(n.bands)].values).index(b) for b in sum(bands, [])]
    groups = [sum(groups, [])[i] for i in indexes]
    l1 = l1.assign_coords({str(n.bgroup): (str(n.bands), groups)})

    if v1_compat: return _v1_compat(l1, filepath)
    return l1.unify_chunks()


def get_sample(level: int = 1) -> Path:
    """
    Retrieve a sample MODIS product file for testing.
    
    Returns paths to pre-configured sample products from environment variables.

    Args:
        level: Processing level (currently only level=1 is supported)

    Returns:
        Path to the MODIS HDF4 file
        
    Raises:
        AssertionError: If the sample file does not exist
    """
    return collect_sample(f'LEVEL{level}_MODISA_MYD', 'nasa', 'AQUA-MODIS-LR', level)
    

################################################################################
# Intern methods
################################################################################

class _Internal:
    
    @staticmethod
    def transform_radiometry(level1: xr.Dataset, chunks: list, resolution: int) -> xr.Dataset:
        """Convert raw DN values to calibrated radiance and reflectance."""
        dim = 'dim'
        
        data_arrays = {'ltoa': [], 'rtoa': []}
        variables = ['EV_250_Aggr1km_RefSB','EV_500_Aggr1km_RefSB','EV_1KM_RefSB','EV_1KM_Emissive']        
        for var in variables:
            
            # Compute radiance
            rad = level1[var]
            bands_dim = only([d for d in rad.dims if str(names.bands) in d])
            bands = level1[bands_dim].values.astype(str)

        # Broadcast scales and offsets with appropriate dimensions
        level1 = level1.assign_coords({new_dim: bands})
        scale = xr.DataArray(rad.radiance_scales, dims=str(names['old'].values))
        offset = xr.DataArray(rad.radiance_offsets, dims=str(names['old'].values))

        nd = scale * rad + offset
        nd.attrs['desc'] = n.ltoa.desc
        nd.attrs['unit'] = n.ltoa.unit
        if resolution:
            nd = nd.rename({str(names['old'].values): str(n.bands)})
        # nd = nd.assign_coords({band_dim: bands})
        
        if resolution:
            data_arrays['ltoa'].append(nd)
        else:
            level1 = level1.assign({str(names['var'].values): nd})

        # Compute Reflectance        
        if any('Emis' in d for d in list(rad.dims)):
            nd = xr.full_like(rad, da.nan)
        
        else:
            # Broadcast scales and offsets with appropriate dimensions
            scale = xr.DataArray(rad.reflectance_scales, dims=str(names['old'].values))
            offset = xr.DataArray(rad.reflectance_offsets, dims=str(names['old'].values))

            nd = scale * rad + offset
            nd.attrs['desc'] = n.rtoa.desc
            nd.attrs['unit'] = n.rtoa.unit
            
        band_dim = str(n.bands) if resolution else str(names['old'].values)
        nd = nd.rename({str(names['old'].values): band_dim})
        nd = nd.assign_coords({band_dim: bands})
        
        if resolution:
            data_arrays['rtoa'].append(nd)
        else:
            level1 = level1.assign({str(names['var'].values): nd})

    # Combine into one dimension
    if resolution:
        level1[str(n.ltoa)] = xr.concat(data_arrays['ltoa'], dim=str(n.bands))
        level1[str(n.ltoa)] = level1[str(n.ltoa)].chunk([1]+list(chunks))
        level1[str(n.rtoa)] = xr.concat(data_arrays['rtoa'], dim=str(n.bands))
        level1[str(n.rtoa)] = level1[str(n.rtoa)].chunk([1]+list(chunks))
        level1 = level1.drop_vars(ds_names['var'].values)
    
    return level1


def _aggregate_vars(l1: xr.Dataset, chunks: list, resolution: int) -> xr.Dataset:
    """Combine uncertainty indices into a single variable."""
    # Combine uncertainty in a single variable
    uncert_label = 'Uncert_Indexes'
    uncert_names = [d for d in l1.variables if uncert_label in d][:4]
    uncert_vars = [l1[v].rename({l1[v].dims[0] : str(n.bands)}) for v in uncert_names]
    
    if resolution:
        l1[uncert_label] = xr.concat(uncert_vars, dim=str(n.bands)).chunk([1]+list(chunks))
        l1 = l1.drop_vars(uncert_names)
    else:
        for v in uncert_vars:
            l1 = l1.assign()
    
    return l1


def _rename_dims(l1: xr.Dataset) -> xr.Dataset:
    """Rename HDF4 dimension names to standard naming convention."""
    revize_dims = {
        '2*nscans:MODIS_SWATH_Type_L1B': str(n.rows)+'_red', 
        '1KM_geo_dim:MODIS_SWATH_Type_L1B': str(n.columns)+'_red', 
        '10*nscans:MODIS_SWATH_Type_L1B': str(n.rows), 
        'Max_EV_frames:MODIS_SWATH_Type_L1B': str(n.columns),
        'new_Band_1KM_Emissive:MODIS_SWATH_Type_L1B': str(n.bands) + '_1KM_Emissive',
        'new_Band_250M:MODIS_SWATH_Type_L1B': str(n.bands) + '_250M',
        'new_Band_500M:MODIS_SWATH_Type_L1B': str(n.bands) + '_500M', 
        'new_Band_1KM_RefSB:MODIS_SWATH_Type_L1B': str(n.bands) + '_1KM_RefSB',
    }
        
    l1 = l1.rename(revize_dims)
    
    # Drop unused dimensions
    coords = l1.coords
    l1 = l1.drop_vars(list(coords))
    l1 = l1.assign_coords({k: v for k,v in coords.items() if k in l1.dims})
    
    return l1

def _compute_bt(ds: xr.Dataset, resolution: int) -> xr.Dataset:
    """Compute brightness temperature from radiance using Planck's law with temperature corrections."""
    # Planck's law constants 
    h = 6.6260755e-34
    c = 2.9979246e+8 # (meters per second)
    k = 1.380658e-23 # (Joules per Kelvin)

    # derived constants
    K1 = 2.0 * h * c * c
    K2 = h * c / k

    # Effective central wavenumber (inverse centimeters)
    cwn = xr.DataArray(da.array([
        2.641775E+3, 2.505277E+3, 2.518028E+3, 2.465428E+3,
        2.235815E+3, 2.200346E+3, 1.477967E+3, 1.362737E+3,
        1.173190E+3, 1.027715E+3, 9.080884E+2, 8.315399E+2,
        7.483394E+2, 7.308963E+2, 7.188681E+2, 7.045367E+2],
        dtype=float), dims=(str(n.bands)))

    # Temperature correction slope (no units)
    tcs = xr.DataArray(da.array([
        9.993411E-1, 9.998646E-1, 9.998584E-1, 9.998682E-1,
        9.998819E-1, 9.998845E-1, 9.994877E-1, 9.994918E-1,
        9.995495E-1, 9.997398E-1, 9.995608E-1, 9.997256E-1,
        9.999160E-1, 9.999167E-1, 9.999191E-1, 9.999281E-1],
        dtype=float), dims=(str(n.bands)))

    # Temperature correction intercept (Kelvin)
    tci = xr.DataArray(da.array([
        4.770532E-1, 9.262664E-2, 9.757996E-2, 8.929242E-2,
        7.310901E-2, 7.060415E-2, 2.204921E-1, 2.046087E-1,
        1.599191E-1, 8.253401E-2, 1.302699E-1, 7.181833E-2,
        1.972608E-2, 1.913568E-2, 1.817817E-2, 1.583042E-2],
        dtype=float), dims=(str(n.bands)))
    
    # Transfer wavenumber [cm^(-1)] to wavelength [m]
    cwvl = 1. / (cwn * 100)
    bands_emis = ds['Band_1KM_Emissive'].values.astype(str)
    offset = xr.Dataset({'cwvl': cwvl, 'tcs': tcs, 'tci': tci})
    offset = offset.assign_coords({str(n.bands): bands_emis})
    
    if resolution:
        bands, var = str(n.bands), str(n.ltoa)
    else :
        bands, var = 'new_Band_1KM_Emissive:MODIS_SWATH_Type_L1B', 'EV_1KM_Emissive'
    
    for b in ds[bands].values:
        array = ds[var].sel({bands: b})
        if b not in offset[str(n.bands)]:
            ds[str(n.bt)+f'_{b}'] = xr.full_like(array, da.nan)
            continue
        off = offset.sel({str(n.bands): b})
        array = K2 / (off['cwvl'] * da.log(K1 / (1e6 * array * off['cwvl'] ** 5) + 1))
        ds[str(n.bt)+f'_{b}'] = ((array - off['tci']) / off['tcs'])
        
    ds = merge(ds, bands, pattern=f'({str(n.bt)})'+r'_(.+)', dtype=str)
    ds[str(n.bt)].attrs['unit'] = 'Kelvin'
    return ds


class _parser_attrs:
    """Parser for MODIS HDF4 metadata attributes."""
    
    def __init__(self, text: list):
        """Initialize parser with metadata text lines."""
        self.data = {}
        self.text = text.copy()
    
    def empty(self) -> bool:
        """Check if there are remaining lines to parse."""
        return len(self.text) == 0
    
    def consume(self) -> str:
        """Remove and return the next non-empty line."""
        
        line = self.text[0]
                
        self.text = self.text[1:]
        
        line = line.strip()
        while self.is_void(line):
            line = self.consume()
        
        return line.strip()
    
    def peek(self) -> str:
        """Return the next non-empty line without consuming it."""
        line = self.text[0].strip()
        while self.is_void(line): 
            self.text = self.text[1:]
            line = self.text[0].strip()
        return line
    
    def is_void(self, line: str) -> bool:
        """Check if a line is empty."""
        return len(line) == 0
    
    def parse(self) -> None:
        """Parse all remaining lines into structured metadata."""
        
        while not self.empty():
            end = self._parse_recu(self.data)
            if end: break
            
    def _parse_recu(self, data: dict = None) -> bool:
        """Recursively parse metadata groups and objects."""
        line = self.consume()
        if line == "END":
            return True
        
        key, val = [i.strip() for i in line.split('=')]
        if key in ['GROUP','OBJECT']: 
            data[val] = {}
            
            line = self.peek()
            while f'END_' not in line:
                self._parse_recu(data[val])
                line = self.peek() # refresh peeked line !! 
            
            # closing tag
            self.consume()
        
        else: data[key] = val 
        
        return False

def get_sample(level: int = 1) -> Path:
    """
    Retrieve a sample MODIS product file for testing.
    
    Returns paths to pre-configured sample products from environment variables.

    Args:
        level: Processing level (currently only level=1 is supported)

    Returns:
        Path to the MODIS HDF4 file
        
    Raises:
        AssertionError: If the sample file does not exist
        
    Example:
        >>> modis_file = get_sample(level=1)
        >>> ds = Level1_MODIS(modis_file)
    """
    sample = env.getdir('DIR_SAMPLE_MODISA_MYD')
    assert sample.exists()
    return sample

def _v1_compat(ds: xr.Dataset, filepath: Path) -> xr.Dataset:
    """Transform dataset to version 1 format for backward compatibility."""
    bands_vis = [650,860,470,555,1240,1640,2130,410,440,485,530,550,668,670,680,685,750,870,900,935,940,1375]
    bands_tir = [3750,3960,4050,4460,4510,1375,6710,7230,8550,9730,11000,12000,13230,13630,13930,14230]
    
    # Drop some variables
    keep = ['latitude','longitude','sza','saa','vza','vaa','Rtoa','BT']
    ds = drop_unused_dims(ds[keep])
    
    # Rename bands IR and change coordinates
    ds = ds.rename({str(n.bands_ir): 'bands_tir'})
    ds = ds.assign_coords(bands_tir=bands_tir)
    
    # Rename Rtoa dimension
    ds = ds.rename({str(n.bands_nvis): 'bands'})
    ds = ds.assign_coords(bands=bands_vis)
    
    # Open reduce latlon arrays
    latlon = load_hdf4(filepath, trim_dims=True)
    ds = ds.assign({str(n.lat): (('y_red','x_red'), latlon['Latitude'].data),
                    str(n.lon): (('y_red','x_red'), latlon['Longitude'].data)})
    
    # Flags
    ds['flags'] = ds['BT'].isel(bands_tir=0).isnull().astype('uint8')
    
    return ds