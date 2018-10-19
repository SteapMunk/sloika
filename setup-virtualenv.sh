#!/bin/bash -ex

source environment

VIRTUALENV_DIR="${DEV_VIRTUALENV_DIR}"
VIRTUALENV_CMD="virtualenv -p python3"

# for fromscratch builds plow through ${VIRTUALENV_DIR} before running this script
mkdir -p ${VIRTUALENV_DIR}
${VIRTUALENV_CMD} ${VIRTUALENV_DIR}

source ${VIRTUALENV_DIR}/bin/activate

python -m pip install pip --upgrade

# install prerequisites of setup.py first
python -m pip install -r scripts/requirements.txt

# install Sloika's dependencies
python -m pip install -r requirements.txt

