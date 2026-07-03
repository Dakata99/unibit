import rich
import rich.console

# set up rich console
console = rich.get_console()

# console print
print = console.print


def rprint(*args, **kwargs):
    if "markup" not in kwargs:
        kwargs["markup"] = True
    print(*args, **kwargs)


def escape(*args):
    return rich.markup.escape(" ".join(map(str, args)))


def colored(color, *args) -> str:
    return f"[{color}]{escape(*args)}[/{color}]"


def info(*args, **kwargs):
    rprint(colored("cyan", "[INFO]"), *args, **kwargs)


def debug(*args, **kwargs):
    rprint(colored("magenta", "[DEBUG]"), *args, **kwargs)
