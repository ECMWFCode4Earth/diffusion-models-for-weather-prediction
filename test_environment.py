import os; print(os.environ["CONDA_PREFIX"])

import torch
import os

from WD.utils import check_devices


############### Check GPU
print(f"The torch version being used is {torch.__version__}")
check_devices()

############### Check Black

# I write ugly code

"""
print ( "Hello World!"        )

"""

# And black makes it better

print("Hello World from Jonathan!")
