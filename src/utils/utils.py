import random

import torch


def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    torch.manual_seed(seed_value)  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
