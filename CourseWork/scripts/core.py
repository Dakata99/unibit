
from plumbum import local
import os
from pathlib import Path
from loguru import logger

CW_ROOT: Path = Path(os.getenv("UNIBIT_CW_ROOT"))

cmake = local["cmake"]
CMAKE_BUILD_DIR: Path = CW_ROOT / 'build'


def configure(*args) -> None:
    logger.info("Configuring CMake project")
    logger.debug(f'CW_ROOT: {CW_ROOT}')
    cmake[
        "-S", CW_ROOT,
        "-G", "Ninja",
        "-B", CMAKE_BUILD_DIR
    ].run_fg()


def build(*args) -> None:
    logger.info("Building CMake project")
    cmake[
        "--build", CMAKE_BUILD_DIR
    ].run_fg()


def run(args):
    logger.info("Running C++ application")

    hcvmain = local[f"{CMAKE_BUILD_DIR}/hcvmain"]
    clargs = [
        f"--AST={args.AST}",
        f"--CHE={args.CHE}",
        f"--ALT={args.ALT}",
        f"--ALP={args.ALP}",
        f"--GGT={args.GGT}",
        args.model,
    ]
    if args.verbose:
        clargs.append("--verbose")
    logger.debug(hcvmain[clargs])
    hcvmain[clargs].run_fg()
