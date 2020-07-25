import torch
import random
import numpy as np
import re
import sys
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def color_print(*s, verbose=1, color=34):
    if verbose < 1:
        return
    colored_str = list()
    color_start_str = "\033[{}m".format(color)
    color_end_str = "\033[0m"
    for text in s:
        color_text = list()
        last_end = 0
        for i in re.finditer(r"\d+\.?\d*", text):
            start, end = i.span()
            color_text.append(text[last_end:start])
            color_text.append(color_start_str)
            color_text.append(text[start:end])
            color_text.append(color_end_str)
            last_end = end
        color_text.append(text[last_end:])
        colored_str.append("".join(color_text))

    sys.stdout.flush()
    if verbose == 1:
        sys.stdout.write("\r" + " ".join(colored_str))
    elif verbose > 1:
        sys.stdout.write("\r" + " ".join(colored_str) + "\n")


def ctqdm(iterable, verbose=1, **kwargs):
    if verbose > 0:
        for i in tqdm(iterable, **kwargs):
            yield i
    else:
        for i in iterable:
            yield i


if __name__ == "__main__":
    color_print("train loss: 123.4234", "test loss: 123.54, time: 12:20:10 s", verbose=2)
