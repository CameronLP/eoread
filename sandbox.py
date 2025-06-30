from eoread.msi import Level1_MSI, get_sample
from core.monitor import Chrono

p = get_sample(1)
with Chrono(unit='s'):
    Level1_MSI(p, resolution='10', chunks=500)