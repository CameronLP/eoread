from tempfile import TemporaryDirectory
from eoread.landsat_oli import Level1_OLI
from eoread.dem import SRTM, GTOPO30
from core.tools import xrcrop


l1_path = '/mnt/ceph/data/LANDSAT9/USA/LC09_L1TP_014034_20220618_20230411_02_T1/'

def test_srtm():
    with TemporaryDirectory() as tmpdir:
        l1 = Level1_OLI(l1_path)
        srtm = SRTM(directory=tmpdir, missing=0)
        sub = xrcrop(srtm, latitude=l1.latitude, longitude=l1.longitude)
        sub.compute(scheduler='sync')

def test_gtopo():
    with TemporaryDirectory() as tmpdir:
        l1 = Level1_OLI(l1_path)
        gtopo = GTOPO30(directory=tmpdir, missing=0)
        sub = xrcrop(gtopo, latitude=l1.latitude, longitude=l1.longitude)
        sub.compute()