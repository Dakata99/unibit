# TODO: some description

## TODO: Prerequisites

- WSL2 (Ubuntu24 only?)
- CMake: `sudo apt install cmake`
- Python: `ln -s /usr/bin/python /usr/bin/python3`

```bash
sudo apt update
sudo apt install build-essential
sudo apt upgrade libstdc++6
```

> NOTE: `libstdc++6` is problematic, when building TFLite source
since its different verion in different distros.

## Formatting/linting

Use `black` to format the python scripts.
Use `ruff` to lint the python scripts.
