#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from sys import argv
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Literal, Union
from core.files.fileutils import filegen
from core.env import getdir


"""
NASA level1 1A or 1B products (MODIS, SeaWiFS, VIIRS) are lacking radiometric
information (polariztion correction). This modules runs l2gen
(https://oceancolor.gsfc.nasa.gov/resources/docs/tutorials/notebooks/oci-ocssw-processing/)
without atmospheric correction, to generate so-called L1C products including the
polarization correction.

l2gen runs either as a standard shell command, or in a docker container.
"""


def makeL1C(
    l1a: Path | str,
    dirname: Path | None = None,
    method: Literal["docker", "shell", "auto"] = "shell",
    **kwargs
) -> Path:
    """
    Generate a L1C product with SeaDAS OCSSW from various sensors

    filename_L1A: path to L1A product (MODIS, SeaWiFS or VIIRS)
    dirname: path to target directory (default None: same directory as l1a)
    method: "shell", "docker" or "auto"
    Other kwargs are passed to l1gen.

    Returns the path to the new product
    """
    l1a = Path(l1a)
    # sensor switch
    if l1a.name.startswith("A"):
        return makeL1C_MODIS(l1a=l1a, dirname=dirname, method=method, **kwargs)
    elif l1a.name.startswith("V") or l1a.name.startswith("JPSS1_VIIRS"):
        return makeL1C_VIIRS(l1a=l1a, dirname=dirname, method=method, **kwargs)
    elif l1a.name.startswith("SEAHAWK1_HAWKEYE"):
        return makeL1C_HAWKEYE(l1a=l1a, dirname=dirname, method=method, **kwargs)
    elif l1a.name.startswith("S"):
        return makeL1C_SeaWIFS(l1a=l1a, dirname=dirname, method=method, **kwargs)
    else:
        raise ValueError(f"Invalid sensor in makeL1C ({l1a.name})")


def makeL1C_HAWKEYE(l1a: Path, dirname: Path | None, method: str, **kwargs) -> Path:
    l1c = (dirname or l1a.parent) / l1a.name.replace("L1A", "L1C")
    with TemporaryDirectory() as tmpdir:
        geo = Path(tmpdir) / (l1a.stem + ".GEO")
        
        if not l1c.exists():
            make_HAWKEYE_GEO(l1a=l1a, geo=geo, method=method)
            assert geo.exists()

        run_l2gen_L1C(ifile=l1a, l1c=l1c, nbands=8, method=method, geofile=geo, **kwargs)

    return l1c

def makeL1C_MODIS(l1a: Path, dirname: Path | None, method: str) -> Path:
    if dirname is None:
        dname = l1a.parent
    else:
        dname = Path(dirname)

    with TemporaryDirectory() as tmpdir:
        geo = Path(tmpdir) / (l1a.stem + ".GEO")
        l1b = Path(tmpdir) / (l1a.stem + ".L1B_LAC")
        l1c = dname / (l1a.stem + ".L1C")

        # Generate GEO and L1B files (temporary)
        if not l1c.exists():
            make_MODIS_GEO(l1a=l1a, geo=geo, method=method)
            assert geo.exists()
            make_L1B_MODIS(l1a=l1a, l1b=l1b, geo=geo, method=method)

        run_l2gen_L1C(
            ifile=l1b,
            l1c=l1c,
            geofile=geo,
            nbands=16,
            method=method,
        )

    return l1c


@filegen(arg="l1b", if_exists="skip")
def make_L1B_MODIS(l1a: Path, l1b: Path, geo: Path, method: str):
    if method == "auto":
        method = get_method_auto("modis_L1B")
    if method == "shell":
        cmd = f"modis_L1B -y -z --okm={l1b} {l1a} {geo}"
        print(cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError("Error in modis_L1B")
        assert l1b.exists()
    else:
        # docker
        cmd = "modis_L1B -y -z --okm={l1b} {l1a}"
        print(cmd)
        app = get_ocssw_docker_app()
        app.run(
            cmd=cmd,
            l1a=l1a,
            l1b=l1b,
        )


@filegen(arg="geo", if_exists="skip")
def make_MODIS_GEO(l1a: Path, geo: Path, method: str):
    if method == "auto":
        method = get_method_auto("modis_GEO")
    if method == "shell":
        cmd = f"modis_GEO --output={geo} {l1a}"
        print(cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError("Error in modis_GEO")
        assert geo.exists()
    else:
        # Docker
        cmd = "modis_GEO --output={geo} {l1a}"
        print(cmd)
        app = get_ocssw_docker_app()
        app.run(
            cmd=cmd,
            l1a=l1a,
            geo=geo,
        )


def makeL1C_VIIRS(l1a: Path, dirname: Path | None, method: str) -> Path:
    if dirname is None:
        dname = l1a.parent
    else:
        dname = Path(dirname)

    with TemporaryDirectory() as tmpdir:

        if str(l1a).endswith(".L1A_SNPP.nc"):
            l1c = dname / (l1a.name.replace(".L1A_SNPP.nc", ".L1C"))
            geo = Path(tmpdir) / (l1a.name.replace(".L1A_SNPP.nc", ".GEO-M_SNPP.nc"))
        elif str(l1a).endswith(".L1A_JPSS1.nc"):
            l1c = dname / (l1a.name.replace(".L1A_JPSS1.nc", ".L1C"))
            geo = Path(tmpdir) / (l1a.name.replace(".L1A_JPSS1.nc", ".GEO-M_JPSS1.nc"))
        elif l1a.name.endswith(".L1A.nc") and l1a.name.startswith("JPSS1_VIIRS"):
            l1c = dname / (l1a.name.replace(".L1A.nc", ".L1C"))
            geo = Path(tmpdir) / (l1a.name.replace(".L1A.nc", ".GEO-M_JPSS1.nc"))
        else:
            raise RuntimeError(f"genL1C_VIIRS: invalid file name {l1a}")
        
        # Generate GEO files (temporary)
        if not l1c.exists():
            make_VIIRS_GEO(l1a=l1a, geo=geo, method=method)
            assert geo.exists()

        run_l2gen_L1C(
            ifile=l1a,
            l1c=l1c,
            geofile=geo,
            nbands=10,
            method=method,
        )

    return l1c


@filegen(arg="geo", if_exists="skip")
def make_VIIRS_GEO(l1a: Path, geo: Path, method: str):
    if method == "auto":
        method = get_method_auto("geolocate_viirs")
    if method == "docker":
        app = get_ocssw_docker_app()
        cmd = "geolocate_viirs ifile={l1a} geofile_mod={geo}"
        print(cmd)
        app.run(
            cmd=cmd,
            l1a=l1a,
            geo=geo,
        )
    else:
        cmd = f"geolocate_viirs ifile={l1a} geofile_mod={geo}"
        print(cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError("Error in genL1C_VIIRS")


@filegen(arg="geo", if_exists="skip")
def make_HAWKEYE_GEO(l1a: Path, geo: Path, method: str):
    if method == "auto":
        method = get_method_auto("geolocate_hawkeye")
    if method == "docker":
        app = get_ocssw_docker_app()
        cmd = "geolocate_hawkeye {l1a} {geo}"
        print(cmd)
        app.run(
            cmd=cmd,
            l1a=l1a,
            geo=geo,
        )
    else:
        cmd = get_prefix()
        cmd += f"geolocate_hawkeye {l1a} {geo}"
        print(cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError("Error in make_HAWKEYE_GEO")


def makeL1C_SeaWIFS(l1a: Path, dirname: Path | None, method: str, **kwargs) -> Path:
    if dirname is None:
        dname = l1a.parent
    else:
        dname = Path(dirname)
    l1c = dname / (l1a.stem + ".L1C")

    run_l2gen_L1C(ifile=l1a, l1c=l1c, nbands=8, method=method, **kwargs)

    return l1c


def get_method_auto(executable: str) -> str:
    if shutil.which(executable) is not None:
        return "shell"
    else:
        return "docker"

def get_ocssw_executable(executable: str) -> str:
    exe = getdir("OCSSWROOT") / "bin" / executable
    assert shutil.which(exe) is not None, (
        f"Could not find {executable}, please check the environment variable OCSSWROOT"
    )
    return str(exe)

def get_prefix() -> str:
    root = getdir("OCSSWROOT")
    cmd = f"OCSSWROOT={root} && source {root}/OCSSW_bash.env && "
    return cmd


def l2gen_cmdline(
    nbands: int,
    geofile: bool,
    **kwargs
) -> str:
    gains = " ".join(["1.0"] * nbands)

    cmd = get_prefix()
    cmd += 'l2gen ifile="{ifile}" ofile="{ofile}" oformat="netcdf4" '
    if geofile:
        cmd += 'geofile="{geofile}" '
    cmd += 'l2prod="rhot_nnn polcor_nnn sena senz sola solz latitude longitude" '
    cmd += f'gain="{gains}" atmocor=0 aer_opt=-99 brdf_opt=0'

    cmd += " " + " ".join([f"{k}={v}" for k, v in kwargs.items()])

    return cmd


def get_ocssw_docker_app(
    version: Union[str, None] = None, tags: Union[list, None] = None
):
    from hydro.apps.OCSSW.OCSSW import OCSSW
    app = OCSSW(version=version, tags=tags)
    return app


@filegen(arg="l1c", if_exists="skip")
def run_l2gen_L1C(
    ifile: Path,
    l1c: Path,
    nbands: int,
    method: str,
    geofile: None | Path = None,
    **kwargs
):
    # run the command
    print("L1A/B:", ifile)
    print("L1C:", l1c)
    if method == "auto":
        method = get_method_auto("l2gen")
    if method == "docker":
        app = get_ocssw_docker_app()
        if geofile is None:
            cmd = l2gen_cmdline(nbands, geofile=False, **kwargs)
            app.run(
                cmd=cmd,
                ifile=ifile,
                ofile=l1c,
            )
        else:
            cmd = l2gen_cmdline(nbands, geofile=True, **kwargs)
            app.run(
                cmd=cmd,
                ifile=ifile,
                ofile=l1c,
                geofile=geofile
            )
        print("CMD:", cmd)

    elif method == "shell":
        if geofile is None:
            cmd = l2gen_cmdline(nbands, geofile=False, **kwargs).format(ifile=ifile, ofile=l1c)
        else:
            cmd = l2gen_cmdline(nbands, geofile=True, **kwargs).format(
                ifile=ifile, ofile=l1c, geofile=geofile
            )
        print("CMD:", cmd)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
        if result.returncode != 0:
            raise RuntimeError(f'Error running command "{cmd}"')
    else:
        raise ValueError(f"Invalid method {method}")


if __name__ == "__main__":
    for l1a in argv[1:]:
        makeL1C(l1a)
