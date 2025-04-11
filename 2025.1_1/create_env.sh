#! /bin/bash

if [ $# != 2 ]; then
    echo "ERROR: Usage:"
    echo "       $0 python3 \$HOME/ccs_venv"
else
    PYTHON=$1
    VENV=$2

    REQUIREMENTS_TXT=$PWD/requirements.txt
    #REQUIREMENTS_TXT=$PWD/requirements-ubuntu-18.04.txt
    #REQUIREMENTS_TXT=$PWD/requirements-centos-7.txt
    #REQUIREMENTS_TXT=$PWD/requirements-sl-7.9.txt

    export PYTHONPATH=

    echo "Creating Python3 Virtual Environment..."
    #$PYTHON -m venv $VENV
    $PYTHON -m venv --without-pip $VENV

    cd $VENV

    echo "Activating Virtual Environment..."
    #    bash
    source $VENV/bin/activate

    echo "Installing pip"
    wget -q -O - https://bootstrap.pypa.io/get-pip.py | python3 -

    echo "Installing packages..."
    pip install --upgrade pip

    pip install -r $REQUIREMENTS_TXT

    export PYTHONPATH=$MGC_HOME/shared/pkgs/ccs_hls4ml/hls4ml
    python3 -c "import hls4ml" 1>/dev/null 2>&1
    EXITCODE=$?
    if [ $EXITCODE -ne 0 ]; then
        echo "HLS4ML install failed"
        exit $EXITCODE
    fi

    echo "========================="
    echo "HLS4ML installed properly"
    echo "========================="

    echo "Done."
    echo "Execute: 'source $VENV/bin/activate' in a new shell to get environment configured"
fi
