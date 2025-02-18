# Catapult 2024.2_2

<!-- vim-markdown-toc GFM -->

* [Tool dependencies](#tool-dependencies)
* [Linux distros](#linux-distros)
* [Setup of the environment variables (to create a Python virtual env)](#setup-of-the-environment-variables-to-create-a-python-virtual-env)
* [Create the Python virtual environment](#create-the-python-virtual-environment)
* [Testing your setup](#testing-your-setup)
* [Known issues](#known-issues)
    - [`cannot find crt*.o: No such file or directory`](#cannot-find-crto-no-such-file-or-directory)
    - [`No QuestaSIM installation found`](#no-questasim-installation-found)
    - [`/share/PLI/MODELSIM/LINUX64/novas_fli.so: cannot open shared object file`](#shareplimodelsimlinux64novas_fliso-cannot-open-shared-object-file)
    - [Issues with `saed32rvt_tt0p78v125c.lib`](#issues-with-saed32rvt_tt0p78v125clib)

<!-- vim-markdown-toc -->

## Tool dependencies

Please check the documentation and dependencies that Catapult requires (`$MGC_HOME/shared/pdfdocs/catapult_relnotes.pdf`). In particular, you need

- Updated license server (SALTD)
- [Siemens Catapult 2024.2_2](https://support.sw.siemens.com/en-US/product/886857312/download/202412051)
- Siemens QuestaSim 2023.2
- Synopsys Novas/Verdi T-2022.06-SP2-1
- Synopsys SAED32nm_PDK_02_2024

This document provides a list of [known issues](#known-issues) due to missing dependencies and their solutions.

## Linux distros

These scripts have been tested with the following Linux distributions:
- Fedora 8.10 ([requirements.txt](requirements.txt))
- AlmaLinux 9.4 ([requirements.txt](requirements.txt))
- Ubuntu 20.04, 22.04 ([requirements.txt](requirements.txt))
- Ubuntu 18.04 ([requirements-ubuntu-18.04.txt](requirements-ubuntu-18.04.txt))
- ScientificLinux 7.9 ([requirements-sl-7.9.txt](requirements-sl-7.9.txt))
- CentOS 7 ([requirements-centos-7.txt](requirements-centos-7.txt))

## Setup of the environment variables (to create a Python virtual env)

To create the Python virtual environment, you **have to** use the Python binaries and libraries provided by Catapult. Thus, you need to export Cataput environment variables and tools. It greatly depends on your local configuration, and you may need to check with your system administrator.

In particular, you need the `MGC_HOME` environment variable defined in your Linux shell. You can check it with
```
echo $MGC_HOME
```
It should point to your local installation of Catapult, for example, `/tools/Siemens/catapult/2024.2_2/Mgc_home`. 

If empty, you can define it, for example:
```
export MGC_HOME=/tools/Siemens/catapult/2024.2_2/Mgc_home # make it sure to use your installation path
```

You also need the directory with Catapult binaries to be at the top of your `PATH` variable:
```
export PATH=${MGC_HOME}/bin:${PATH}
```

Finally, check if the Python executable comes from the Catapult installation directory:
```
which python3
```
That should return something like `/tools/Siemens/catapult/2024.2_2/Mgc_home/bin/python3`; if instead, it is `/usr/bin/python3`, then either `PATH` or `MGC_HOME` are not properly set.

## Create the Python virtual environment

Edit `create_env.sh` given the Linux distribution. You can comment/uncomment the correct requirements file given the distribution you are running (see beginning of the page).
```
REQUIREMENTS_TXT=$PWD/requirements.txt
#REQUIREMENTS_TXT=$PWD/requirements-ubuntu-18.04.txt
#REQUIREMENTS_TXT=$PWD/requirements-centos-7.txt
#REQUIREMENTS_TXT=$PWD/requirements-sl-7.9.txt
```

Then, run the following script:
```
./create_env.sh python3 $HOME/ccs_venv
```

Creating the Python virtual environment should take several minutes. On success:
```
=========================
HLS4ML installed properly
=========================

Done.
Execute: 'source $VENV/bin/activate' in a new shell to get environment configured
```

If the process fails, you will get:
```
HLS4ML install failed
```

## Testing your setup

Make sure the following tools and libraries are installed besides Catapult 2024.2_2:
- Siemens QuestaSim 2023.2
- Synopsys Novas/Verdi T-2022.06-SP2-1
- Synopsys SAED32nm_PDK_02_2024

Activate the Python environment previously created:
```
source $HOME/ccs_venv/bin/activate
```

Run one of the hls4ml / Catapult AI NN examples provided with Catapult AI NN.

First:
```
PYTHONPATH=$MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml python $MGC_HOME/shared/examples/cat_ai_nn/Simple/model_asic.py --reuse_factor=1
```

Then:
```
cd my-Catapult-test1; catapult -shell -file build_prj.tcl
```

If you notice any errors, please check the following solutions to known issues.

## Known issues

### `cannot find crt*.o: No such file or directory`

If you get the following (or similar) error when running an `hls4ml` design in Catapult AI NN
```
/usr/bin/ld: cannot find crt1.o: No such file or directory
```

Add the variable `LIBRARY_PATH` to the file `$MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml/hls4ml/templates/catapult/build_lib.sh`:
```
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    ${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.o ${PROJECT}_bridge.o -o firmware/${PROJECT}-${LIB_STAMP}.so

# and a little below...

LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    ${CC} -std=c++11 -g ${INCFLAGS} firmware/${PROJECT}.cpp ${PROJECT}_test.cpp -o ${PROJECT}.exe
```

### `No QuestaSIM installation found`

If you get the following error when running an `hls4ml` design in Catapult AI NN
```
# $MGC_HOME/shared/include/mkfiles/ccs_questasim.mk:164: *** No QuestaSIM installation found. Check your QuestaSIM flow Path option in Catapult.  Stop.
```

Make sure that QuestaSIM (`vsim`) is in the path:
```
export MODEL_TECH=PATH_TO_QUESTASIM/bin
export PATH=$MODEL_TECH:$PATH
```

If you are running Catapult in the GUI model, you can configure `Tools >> Set Options... >> Flows >> QuestaSIM >> Path` to point to the `bin` directory in the QuestaSIM installation.

### `/share/PLI/MODELSIM/LINUX64/novas_fli.so: cannot open shared object file`

If you get the following error when running an `hls4ml` design in Catapult AI NN
```
# Loading /share/PLI/MODELSIM/LINUX64/novas_fli.so
# Error: (suppressible): (vsim-3197) Load of "/share/PLI/MODELSIM/LINUX64/novas_fli.so" failed: /share/PLI/MODELSIM/LINUX64/novas_fli.so: cannot open shared object file: No such file or directory.
# Error: (suppressible): (vsim-PLI-3002) Failed to load PLI object file "/share/PLI/MODELSIM/LINUX64/novas_fli.so".
# Time: 0 ps  Iteration: 0  Root: /
# Error: loading design
# Error: loading design
# End time: 23:18:28 on Oct 21,2024, Elapsed time: 0:00:01
# Errors: 2, Warnings: 0, Suppressed Warnings: 1
# Error: make: *** [sim] Error 12
# Info: SCVerify complete for use mode default
# Error: Switching Activity File 'my-Catapult-test_asic1/myproject_prj/myproject.v1/switching_vhdl/default.fsdb' was not found after simulation
# Error:      The lack of a file is usually caused by compile errors in SCVerify
# Info: Completed transformation 'prototyping' on solution 'myproject.v1': elapsed time 13.60 seconds, memory usage 1572860kB, peak memory usage 1572924kB (SOL-9)
# Info: Design complexity at end of 'prototyping': Total ops = 1513, Real ops = 609, Vars = 621 (SOL-21)
# Error: errors encountered while running flows, check $errorInfo for more details
# Error: Switching Activity File not found after simulation
# Error: Package: PowerAnalysis, Version: 8.0a, Flow: switching
# Error: Script: $MGC_HOME/pkgs/sif/userware/En_na/flows/app_poweranalysis.flo ...
```

Make sure that Novas/Verdi variable is defined:
```
export VERDI_HOME=PATH_TO_NOVAS_VERDI
```

If you are running Catapult in the GUI model, you can configure `Tools >> Set Options... >> Flows >> Novas >> Path` to point to the Novas Verdi installation directory.

### Issues with `saed32rvt_tt0p78v125c.lib`

If you get an error and Catapult cannot find `saed32rvt_tt0p78v125c.lib`:

1. Make sure you have installed the _Synopsys SAED32nm PDK_;
2. Make sure you have properly exported the `SAED32_EDK` environment variable:
   ```
   export SAED32_EDK=PATH_TO_SYNOPSYS_SAED32_EDK
   ```
3. At _line 28_ of the script `$MGC_HOME/pkgs/siflibs/saed/setup_saedlib.tcl` add:
   ```
   options set ComponentLibs/TechLibSearchPath [file join $::env(SAED32_EDK) lib stdcell_rvt SAED32_EDK lib stdcell_rvt db_ccs] -append   
   ```
