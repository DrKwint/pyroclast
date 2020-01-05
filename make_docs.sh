#!/bin/bash

if [ -x "$(command -v grealpath)" ]; then
    # MacOS
    cd "$(dirname $(grealpath "$0"))"
else
    # Linux
    cd "$(dirname $(realpath "$0"))"
fi

sphinx-apidoc -f -o docs/api pyroclast
cd docs
make html
