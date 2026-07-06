# Bootstrap

## How to install WSL?

1. Get WSL2 (Ubuntu24)

    Follow this guide: https://learn.microsoft.com/en-us/windows/wsl/install

    - Install Ubuntu 24 in WSL2

        1.1. Open Microsoft Store.

        1.2. Search for Ubuntu 24.04 LTS.

        1.3. Click Install.

        1.4. Once installed, launch it from Start Menu or run:
        ```bash
        wsl -d Ubuntu-24.04
        ```

    After installation, you need to set up a username and password. Then run:
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

2. Install needed dependencies
```bash
sudo apt install python3-venv python-is-python3 build-essential cmake ninja-build
```
