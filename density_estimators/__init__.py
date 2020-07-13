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
