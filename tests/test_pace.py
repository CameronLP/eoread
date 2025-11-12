from pathlib import Path
from matplotlib import pyplot as plt
import pytest
from eoread.pace import Level1B_PACE_OCI, get_sample
import xarray as xr
from core.tests import conftest



@pytest.fixture
def level1_oci() -> Path:
    return get_sample()['path']


def test_read_pace_oli(level1_oci, request):
    l1 = Level1B_PACE_OCI(level1_oci)

    xr.set_options(display_max_rows=200)

    print(l1)

    plt.imshow(
        l1['Rtoa'].sel(bands=865, method='nearest'),
        vmin=0, vmax=1)
    plt.colorbar()

    conftest.savefig(request)




