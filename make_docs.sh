#!/bin/bash

cd "$(dirname $(realpath "$0"))"
sphinx-apidoc -f -o docs/source projectdir
cd docs
make html
