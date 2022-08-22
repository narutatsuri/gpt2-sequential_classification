from util import *
from util.functions import *
import pandas as pd


def load_dataset():
    """
    """
    df = pd.read_csv(dataset_dir)
    neg = list(df[df.columns[1]])
    pos = list(df[df.columns[2]])
    
    X = neg + pos
    y = [0] * len(neg) + [1] * len(pos)
    
    return {"X": X, "y": y}