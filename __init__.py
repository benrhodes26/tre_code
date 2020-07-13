import os
import sys

local_pc_root = os.path.expanduser("~/tre_code/")
if os.path.isdir(local_pc_root):
    project_root = local_pc_root
else:
    project_root = os.path.expanduser("~/")

density_data_root = os.path.expanduser(project_root + 'density_data/')
utils_root = os.path.expanduser(project_root + 'utils/')

roots = [project_root, density_data_root, utils_root]
for root in roots:
    if root not in sys.path:
        sys.path.append(root)
