import argparse
import argcomplete
from loguru import logger

EPOCHS: int = 50
BATCH_SIZE: int = 16

def setup_logging(debug: bool = False) -> None:
    import sys

    from loguru import logger

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if debug else "INFO",
    )

def main():
    parser = argparse.ArgumentParser(prog="hcv")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    subparsers = parser.add_subparsers(dest="command")

    # hcv configure
    _ = subparsers.add_parser("configure", help="Configure CMake project.")

    # hcv build
    _ = subparsers.add_parser("build", help="Build the C++ application.")

    # hcv run
    run_parser = subparsers.add_parser("run", help="Run the C++ application.")
    run_parser.add_argument("--AST", required=True, help="AST value")
    run_parser.add_argument("--CHE", required=True, help="CHE value")
    run_parser.add_argument("--ALT", required=True, help="ALT value")
    run_parser.add_argument("--ALP", required=True, help="ALP value")
    run_parser.add_argument("--GGT", required=True, help="GGT value")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbosity.")
    run_parser.add_argument("model", help="TFLite model to run")

    # hcv train
    train_parser = subparsers.add_parser("train", help="Run C++ application.")
    train_parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of epochs to train the model."
    )
    train_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    train_parser.add_argument("--convert", action="store_true", help="Convert model to TFLite.")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.debug)

    logger.debug(f"Args: {args}")

    try:
        if args.command == 'train':
            from .train import main as train

            train(args.epochs, args.batch_size, args.convert)
        else:
            from .core import build, configure, run

            commands = {
                "configure": configure,
                "build": build,
                "run": run
            }
            commands[args.command](args)
    except Exception as e:
        logger.error(f'Failed with exception: {e}')
