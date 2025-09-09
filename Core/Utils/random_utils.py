import os
import random
from typing import Optional, NoReturn

import numpy as np
import torch

# (Basic) Fix the random seed to make results reproducible
def fix_random_seed(seed: Optional[int] = 927) -> NoReturn:
    if seed is None:
        return

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

# (Advanced) Fix the random seed to make results reproducible
def fix_all_random_seed(seed: Optional[int] = 927) -> NoReturn:
    if seed is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    os.environ['PYTHONHASHSEED'] = str(1)
