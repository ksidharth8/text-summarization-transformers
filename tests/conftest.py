import os
import sys

# make `import src` work when pytest is launched from any directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
