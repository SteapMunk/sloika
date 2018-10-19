#!/bin/bash -ex

source environment

VIRTUALENV_DIR="${DEV_VIRTUALENV_DIR}"

source ${VIRTUALENV_DIR}/bin/activate

python -m pip install pip --upgrade

python -m pip install -r misc/requirements.txt

python -m pip install -r setup-dev-env.txt

python setup.py develop
