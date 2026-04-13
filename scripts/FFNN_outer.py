#!/usr/bin/python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from time import time, sleep
from datetime import datetime

from paths import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="FFNN Collection Script")