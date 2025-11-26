import argparse
from plumbum import local

from scripts import mylogging as log

cmake = local["cmake"]


def configure(args):
    log.info("Configuring CMake project")
    args = ["-S", ".", "-B", "build"]
    cmake[args].run_fg()


def build(args):
    log.info("Building CMake project")
    args = ["--build", "build", "-j", 6]
    cmake[args].run_fg()


def run(args):
    log.info("Running C++ application")
    log.debug("Args: ", args)
    hcvmain = local["./build/hcvmain"]
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
    log.debug(hcvmain[clargs])
    hcvmain[clargs].run_fg()


def lint(args):
    scripts = ["dl.py", "train.py", "mylogging.py"]

    # First format the files with `black`
    cmd = local["black"]
    cmd[scripts].run_fg()

    # Then lint the files with `ruff`
    cmd = local["ruff"]["check", "--fix"]
    cmd[scripts].run_fg()


commands = {"configure": configure, "build": build, "run": run, "lint": lint}


def main():
    parser = argparse.ArgumentParser(prog="dl")
    subparsers = parser.add_subparsers(dest="command")

    # dl configure
    _ = subparsers.add_parser("configure", help="Configure CMake project")

    # dl build
    _ = subparsers.add_parser("build", help="Build C++ application")

    # dl run
    run_parser = subparsers.add_parser("run", help="Run C++ application")
    run_parser.add_argument("AST", help="AST description")
    run_parser.add_argument("CHE", help="CHE description")
    run_parser.add_argument("ALT", help="ALT description")
    run_parser.add_argument("ALP", help="ALP description")
    run_parser.add_argument("GGT", help="GGT description")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity")
    run_parser.add_argument("model", help="TFLite model to run")

    # dl lint (+ format)
    _ = subparsers.add_parser("lint", help="Lint (+ format) Python scripts")

    args = parser.parse_args()
    commands[args.command](args)


if __name__ == "__main__":
    main()
