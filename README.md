# Workstation Setup for Siemens Catapult AI NN

When installing [Siemens Catapult AI NN](https://newsroom.sw.siemens.com/en-US/siemens-catapult-ai-nn/) your main reference should be the official documentation on the [Siemens software download portal](https://support.sw.siemens.com/en-US/product/886857312/downloads) or in your local installation directory (`$MGC_HOME`). Please refer to:

- `$MGC_HOME/shared/pdfdocs/catapult_install.pdf`
- `$MGC_HOME/shared/pdfdocs/catapult_relnotes.pdf`
  - In particular, check the Software Compatibility section.

In addition to the official support, this repository provides help for setting up Catapult (AI NN), including scripts for creating Python virtual environments and instructions to apply hotfixes.

A Python virtual environment is typically created once, the first time you run a Catapult AI NN project. By default, Catapult AI NN places the environment in your home directory under the `ccs_venv` subdirectory (`$HOME/ccs_venv`). The default scripts may fail on some Linux distributions, and this repository provides fixes for those cases.

The scripts create a distro-specific Python virtual environment for Catapult AI NN (`$HOME/ccs_venv`). Each directory corresponds to a specific Catapult release, such as `2024.2_2`, and some directories include configuration files for different Linux distributions. Please take a look at the README file in each directory for details.

You can use `diff` to compare the contents of these directories and monitor changes between releases. Additionally, you can reference the original scripts provided by Siemens in your Catapult installation directory at `$MGC_HOME/shared/pkgs/ccs_hls4ml/create_env.sh`.

Finally, solutions are provided for the most common issues you may encounter.
