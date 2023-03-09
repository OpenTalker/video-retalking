#!/bin/bash

rm -rf docs
# reStructuredText in python files to rst. Documentation in docs folder
sphinx-apidoc -A "Alyssa Quek" -f -F -o docs facemorpher/

cd docs

# Append module path to end of conf file
echo "" >> conf.py
echo "import os" >> conf.py
echo "import sys" >> conf.py
echo "sys.path.insert(0, os.path.abspath('../'))" >> conf.py
echo "sys.path.insert(0, os.path.abspath('../facemorpher'))" >> conf.py

# Make sphinx documentation
make html
cd ..
