from eoread.tools import open_raster, filter_metadata, spatial_resample
from core.tests.graphics import xrimshow, downsample

from warnings import warn
warn('The module `utils` will be deprecated from eoread package and replace by `tools` module.', DeprecationWarning)