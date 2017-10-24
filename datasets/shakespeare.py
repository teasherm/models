import os
import sys
import urllib.request

import numpy as np

resource_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load(data_file="data/shakespeare/input.txt"):
  if not os.path.exists(os.path.dirname(data_file)):
    os.makedirs(os.path.dirname(data_file))
  if not os.path.exists(data_file):
    sys.stdout.write("Downloading shakespeare corpus...")
    resp = urllib.request.urlopen(resource_url)
    with open(data_file, "wb") as f:
      f.write(resp.read())
    print("done")
  f =  open(data_file, "r")
  txt = f.read()
  f.close()
  char2idx = {char: i for i, char in enumerate(set(txt))}
  idx2char = {i: char for i, char in enumerate(set(txt))}
  X = np.array([char2idx[x] for x in txt])
  y = [char2idx[x] for x in txt[1:]]
  y.append(char2idx['.'])
  y = np.array(y)
  return X, y, char2idx, idx2char

