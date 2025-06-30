from eoread.utils import filter_metadata, spatial_resample
from eoread.hdf4 import load_hdf4
from pathlib import Path

from core.geo import n
from core import env, log
from core.tools import drop_unused_dims

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


def Level1_MODIS(filepath: Path | str, 
                 chunks: int = 100,
                 metadata_template: list = None,
                 v1_compat: bool = False):
    
    filepath = Path(filepath)
    assert filepath.exists(), 'File does not exists'
    if isinstance(chunks, int): chunks = [chunks]*2   
    
    # Revize variables
    log.debug('Reading h4file')
    l1 = load_hdf4(filepath, trim_dims=True)
    l1 = l1.rename_vars({
        'Latitude': n.lat.name, 'Longitude': n.lon.name,
        'SensorZenith': n.vza.name, 'SensorAzimuth': n.vaa.name,
        'SolarZenith': n.sza.name, 'SolarAzimuth': n.saa.name
    })
             
    # Read metadata
    metadata = {}
    log.debug('parsing metadata text')
    for name in ['CoreMetadata.0','ArchiveMetadata.0','StructMetadata.0']:
        p = parser_attrs(l1.attrs[name].split('\n'))
        p.parse()
        metadata.update(p.data)
    
    # Add band information
    log.debug('Add central wavelength')
    l1 = l1.assign({n.bnames.name: ((n.bands.name), da.array(bnames).astype(str)),
                    n.cwav.name: ((n.bands.name), cwvl)})
    
    # Rescale angles data
    log.debug('Read and compute geometric angles')
    for varname in [n.vza.name, n.vaa.name, n.sza.name, n.saa.name]:
        l1[varname] = l1[varname].scale_factor * l1[varname]

    # Change radiometry of input data   
    log.debug('Read top of atmosphere data')
    l1 = transform_radiometry(l1, chunks)
    l1 = aggregate_vars(l1, chunks)
    # l1 = compute_bt(l1)
    
    # Upscale latlon variables
    log.debug('upscale latlon variables')
    shape = {k:v for k,v in zip(l1[n.lat.name].dims, l1[n.ltoa.name].shape[1:])}
    l1[n.lat.name] = spatial_resample(l1[n.lat.name], shape, chunks)
    l1[n.lon.name] = spatial_resample(l1[n.lon.name], shape, chunks)

    # Change dimensions name and update coordinates
    l1 = rename_dims(l1)

    # Summarize Attributes
    log.debug('Add important attributes')
    attributes = l1.attrs
    
    l1.attrs = {}
    l1.attrs[n.input_directory.name] = str(filepath.parent)
    l1.attrs[n.resolution.name]   = 1000
    l1.attrs[n.datetime.name]     = metadata['INVENTORYMETADATA']['ECSDATAGRANULE']['PRODUCTIONDATETIME']['VALUE'][1:-1]
    l1.attrs['night']             = str(metadata['INVENTORYMETADATA']['ECSDATAGRANULE']['DAYNIGHTFLAG']['VALUE'] != '"Day"')
    l1.attrs[n.product_name.name] = metadata['INVENTORYMETADATA']['ECSDATAGRANULE']['LOCALGRANULEID']['VALUE'][1:-1]
    l1.attrs[n.platform.name]     = metadata['INVENTORYMETADATA']['ASSOCIATEDPLATFORMINSTRUMENTSENSOR']['ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']['ASSOCIATEDPLATFORMSHORTNAME']['VALUE'][1:-1]
    l1.attrs[n.sensor.name]       = metadata['INVENTORYMETADATA']['ASSOCIATEDPLATFORMINSTRUMENTSENSOR']['ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']['ASSOCIATEDSENSORSHORTNAME']['VALUE'][1:-1]
    l1.attrs[n.shortname.name]    = metadata['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE'][1:-1]
    l1.attrs['version']           = int(metadata['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['VERSIONID']['VALUE'])
    l1.attrs['user_guide']        = user_guide

    filter_fn = (lambda x,y: x) if metadata_template is None else filter_metadata
    metadata['attributes'] = attributes
    l1.attrs['metadata'] = filter_fn(metadata, metadata_template)

    return drop_unused_dims(l1).unify_chunks()


def transform_radiometry(level1, chunks):
    
    rad_varnames = ['EV_250_Aggr1km_RefSB','EV_500_Aggr1km_RefSB','EV_1KM_RefSB','EV_1KM_Emissive'] #TOA
    rad_bandnames = ["new_Band_250M:MODIS_SWATH_Type_L1B", 
                     "new_Band_500M:MODIS_SWATH_Type_L1B", 
                     "new_Band_1KM_RefSB:MODIS_SWATH_Type_L1B",
                     "new_Band_1KM_Emissive:MODIS_SWATH_Type_L1B"]
    
    # Compute radiance
    data_arrays = []
    for rad_varname, rad_bandname in zip(rad_varnames, rad_bandnames):
        rad = level1[rad_varname]

        # Broadcast scales and offsets with appropriate dimensions
        scale = xr.DataArray(rad.radiance_scales, dims=rad_bandname)
        offset = xr.DataArray(rad.radiance_offsets, dims=rad_bandname)

        # Expand scale/offset to match rad's dims for broadcasting
        scale = scale.broadcast_like(rad)
        offset = offset.broadcast_like(rad)

        nd = scale * rad + offset
        nd.attrs['desc'] = n.ltoa.desc
        nd.attrs['unit'] = n.ltoa.unit
        data_arrays.append(nd.rename({rad_bandname : "bands_ltoa"}))

    # Combine into one dimension
    level1[n.ltoa.name] = xr.concat(data_arrays, dim="bands_ltoa")
    level1[n.ltoa.name] = level1[n.ltoa.name].chunk([1]+list(chunks))
    
    # Compute Reflectance
    data_arrays = []
    for rad_varname, rad_bandname in zip(rad_varnames[:3], rad_bandnames[:3]):
        rad = level1[rad_varname]

        # Broadcast scales and offsets with appropriate dimensions
        scale = xr.DataArray(rad.reflectance_scales, dims=rad_bandname)
        offset = xr.DataArray(rad.reflectance_offsets, dims=rad_bandname)

        # Expand scale/offset to match rad's dims for broadcasting
        scale = scale.broadcast_like(rad)
        offset = offset.broadcast_like(rad)

        nd = scale * rad + offset
        nd.attrs['desc'] = n.rtoa.desc
        nd.attrs['unit'] = n.rtoa.unit
        data_arrays.append(nd.rename({rad_bandname : "bands_rtoa"}))

    # Combine into one dimension
    level1[n.rtoa.name] = xr.concat(data_arrays, dim="bands_rtoa")
    level1[n.rtoa.name] = level1[n.rtoa.name].chunk([1]+list(chunks))
    
    return level1.drop_vars(rad_varnames)


def aggregate_vars(l1, chunks):
    
    # Combine uncertainty in a single variable
    uncert_label = 'Uncert_Indexes'
    uncert_names = [d for d in l1.variables if uncert_label in d][:4]
    uncert_vars = [l1[v].rename({l1[v].dims[0] : "b"}) for v in uncert_names]
    l1[uncert_label] = xr.concat(uncert_vars, dim="b").chunk([1]+list(chunks))
    l1 = l1.drop_vars(uncert_names)
    
    return l1.rename(b=n.bands.name)


def rename_dims(l1):
    
    revize_dims = {
        '2*nscans:MODIS_SWATH_Type_L1B': n.rows.name+'_red', 
        '1KM_geo_dim:MODIS_SWATH_Type_L1B': n.columns.name+'_red', 
        '10*nscans:MODIS_SWATH_Type_L1B': n.rows.name, 
        'Max_EV_frames:MODIS_SWATH_Type_L1B': n.columns.name,
        'new_Band_1KM_Emissive:MODIS_SWATH_Type_L1B': n.bands_ir.name,
        'new_Band_250M:MODIS_SWATH_Type_L1B': n.bands.name + '_250M',
        'new_Band_500M:MODIS_SWATH_Type_L1B': n.bands.name + '_500M', 
        'new_Band_1KM_RefSB:MODIS_SWATH_Type_L1B': n.bands.name + '_1KM_NVIS',
        'bands_ltoa': n.bands.name, 
        'bands_rtoa': n.bands_nvis.name, 
        'Band_250M': n.bands.name + '_250M', 
        'Band_500M': n.bands.name + '_500M', 
        'Band_1KM_RefSB': n.bands.name + '_1KM_NVIS',
        'Band_1KM_Emissive': n.bands.name + '_1KM_EM',
    }

    return l1.rename(revize_dims)

def compute_bt(ds):
    """Calibration for the emissive channels."""

    # Planck's law constants 
    h = 6.6260755e-34
    c = 2.9979246e+8 # (meters per second)
    k = 1.380658e-23 # (Joules per Kelvin)

    # derived constants
    K1 = 2.0 * h * c * c
    K2 = h * c / k
    bands = 'bands_ltoa'

    # Effective central wavenumber (inverse centimeters)
    cwn = xr.DataArray(da.array([
        2.641775E+3, 2.505277E+3, 2.518028E+3, 2.465428E+3,
        2.235815E+3, 2.200346E+3, 1.477967E+3, 1.362737E+3,
        1.173190E+3, 1.027715E+3, 9.080884E+2, 8.315399E+2,
        7.483394E+2, 7.308963E+2, 7.188681E+2, 7.045367E+2],
        dtype=float), dims=(bands))

    # Temperature correction slope (no units)
    tcs = xr.DataArray(da.array([
        9.993411E-1, 9.998646E-1, 9.998584E-1, 9.998682E-1,
        9.998819E-1, 9.998845E-1, 9.994877E-1, 9.994918E-1,
        9.995495E-1, 9.997398E-1, 9.995608E-1, 9.997256E-1,
        9.999160E-1, 9.999167E-1, 9.999191E-1, 9.999281E-1],
        dtype=float), dims=(bands))

    # Temperature correction intercept (Kelvin)
    tci = xr.DataArray(da.array([
        4.770532E-1, 9.262664E-2, 9.757996E-2, 8.929242E-2,
        7.310901E-2, 7.060415E-2, 2.204921E-1, 2.046087E-1,
        1.599191E-1, 8.253401E-2, 1.302699E-1, 7.181833E-2,
        1.972608E-2, 1.913568E-2, 1.817817E-2, 1.583042E-2],
        dtype=float), dims=(bands))

    # Transfer wavenumber [cm^(-1)] to wavelength [m]
    cwvl = 1. / (cwn * 100)

    # Some versions of the modis files do not contain all the bands
    bnames = list(ds[n.bnames.name].data.astype(float).tolist())
    bands_em = [bnames.index(x) for x in ds['Band_1KM_Emissive']]
    array = ds[n.ltoa.name].sel({bands: bands_em})
    array = K2 / (cwvl * da.log(K1 / (1e6 * array * cwvl ** 5) + 1))
    ds[n.bt.name] = ((array - tci) / tcs).rename({bands: n.bands_ir.name})
    return ds


class parser_attrs:
    
    def __init__(self, text: list):
        self.data = {}
        self.text = text.copy()
    
    def empty(self): return len(self.text) == 0
    
    def consume(self):
        
        line = self.text[0]
                
        self.text = self.text[1:]
        
        line = line.strip()
        while self.is_void(line):
            line = self.consume()
        
        return line.strip()
    
    def peek(self):
        line = self.text[0].strip()
        while self.is_void(line): 
            self.text = self.text[1:]
            line = self.text[0].strip()
        return line
    
    def is_void(self, line):
        return len(line) == 0
    
    def parse(self):
        
        while not self.empty():
            end = self._parse_recu(self.data)
            if end: break
            
    def _parse_recu(self, data: dict=None): 
        
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

def get_sample(level: int=1, use_cache:bool=True):
    sample = Path('/mnt/ceph/data/MODIS_AQUA/MYD021KM.A2016010.0150.006.2016012022653.hdf')
    assert sample.exists()
    return sample