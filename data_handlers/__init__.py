import os
import sys

default_path = os.path.expanduser("~/tre_code/")
if os.path.isdir(default_path):
    project_root = default_path
else:
    project_root = os.path.expanduser("~/")

density_data_root = os.path.expanduser(project_root + 'density_data/')

roots = [project_root, density_data_root]
for root in roots:
    if root not in sys.path:
        sys.path.append(root)

from data_handlers.gaussians import GAUSSIANS
from data_handlers.mnist import MNIST
from data_handlers.grid_data import *
from data_handlers.one_dim_data import *
from data_handlers.multi_omniglot import MultiOmniglot
