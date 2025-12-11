# OpenFGL/openfgl/flcore/hcfl_plus/__init__.py
import os
import sys

# add HCFL repo to python path; adjust relative path if your layout differs
HERE = os.path.dirname(__file__)
HCFL_PATH = os.path.abspath(os.path.join(HERE, "../../../../HCFL"))

if HCFL_PATH not in sys.path:
    sys.path.insert(0, HCFL_PATH)
