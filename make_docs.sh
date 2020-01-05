#!/bin/bash

if [ -x "$(command -v grealpath)" ]; then
    # MacOS
    cd "$(dirname $(grealpath "$0"))"
else
    # Linux
    cd "$(dirname $(realpath "$0"))"
fi

sphinx-apidoc -f -o docs/api pyroclast pyroclast/svae/* pyroclast/selfboosting/* pyroclast/qualia/*
cd docs
make html
