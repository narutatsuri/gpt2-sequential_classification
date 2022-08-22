from util import * 
from util.functions import *
from util.models import *
import sys


# Load dataset
dataset = load_dataset()

model = GPT2_classifier(dataset)
model.train()