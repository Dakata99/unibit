
from plumbum import local
import os
from pathlib import Path
from loguru import logger

CW_ROOT: Path = Path(os.getenv("UNIBIT_CW_ROOT") or f'{os.getenv("UNIBIT_ROOT")}/CourseWork')

cmake = local["cmake"]
CMAKE_BUILD_DIR: Path = CW_ROOT / 'build'
HCVMAIN: Path = CMAKE_BUILD_DIR / 'hcvmain'


def configure(*args) -> None:
    logger.info("Configuring CMake project")
    logger.debug(f'CMAKE_BUILD_DIR: {CMAKE_BUILD_DIR}')
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

    hcvmain = local[HCVMAIN]
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
