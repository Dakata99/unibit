import argparse
import argcomplete
from loguru import logger

def main():
    parser = argparse.ArgumentParser(prog="hcv")
    subparsers = parser.add_subparsers(dest="command")

    # hcv configure
    _ = subparsers.add_parser("configure", help="Configure CMake project")

    # hcv build
    _ = subparsers.add_parser("build", help="Build C++ application")

    # hcv run
    run_parser = subparsers.add_parser("run", help="Run C++ application")
    run_parser.add_argument("--AST", required=True, help="AST value")
    run_parser.add_argument("--CHE", required=True, help="CHE value")
    run_parser.add_argument("--ALT", required=True, help="ALT value")
    run_parser.add_argument("--ALP", required=True, help="ALP value")
    run_parser.add_argument("--GGT", required=True, help="GGT value")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    run_parser.add_argument("model", help="TFLite model to run")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    logger.debug(f"Args: {args}")

    from .core import build, configure, run

    commands = {
        "configure": configure,
        "build": build,
        "run": run
    }
    try:
        commands[args.command](args)
    except Exception as e:
        logger.error(f'Failed with exception: {e}')
