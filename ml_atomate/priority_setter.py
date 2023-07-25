import argparse
import sys
import copy
from collections import defaultdict
from datetime import datetime
from itertools import count, chain
from typing import Dict, List
from pathlib import Path
from atomate.utils.utils import get_database
import random
import json
import os
import time
import traceback
from math import log10
from logging import getLogger, DEBUG, basicConfig
import numpy as np
import pandas as pd
import multiprocessing
from pymongo import database
from fireworks import LaunchPad
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

from ml_atomate.physbo_customized.policy_ptr import Policy
from ml_atomate.utils.util import get_from_mongo_like_str, parse_objective
from ml_atomate.blox_kterayama.curiosity_sampling import stein_novelty
import importlib


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_file", "-df",
                        help="path to db.json",
                        type=str,
                        required=True)

    parser.add_argument("--builder", "-bld",
                        help="Specify python file containing run_builder function",
                        type=str)

    parser.add_argument("--descriptor_csv", "-dc",
                        help="path to descriptor.csv",
                        type=str,
                        required=True)

    parser.add_argument("--objective", "-o",
                        help="Set prediction objective. "
                             "When you use PTR function,"
                             "you can use mongolike specification (e.g. other_prop.energy_per_atom)"
                             "Write range together, like "
                             "bandstructure_hse.bandgap 4.0, dielectric.epsilon_avg 30.0,",
                        type=str,
                        nargs="+",
                        required=True
                        )

    parser.add_argument("--conversion", "-c",
                        help="Select no_conversion or log",
                        type=str,
                        nargs="+",
                        )

    parser.add_argument("--property_descriptor", "-pd",
                        help="Set property when you want to use other property as descriptor, e.g. GGA band_gap.",
                        type=str,
                        nargs="+",
                        )

    parser.add_argument("--n_estimators", "-e",
                        help="The number of trees "
                             "for the Random Forest Regression",
                        type=int,
                        default=1000)

    parser.add_argument("--n_seeds", "-ns",
                        help="The number of seeds for CV",
                        type=int,
                        default=1)

    parser.add_argument("--n_cv_folds", "-n",
                        help="The N value for N-folds cross validation",
                        type=int,
                        default=0)

    parser.add_argument("--permutation_importance", "-pi",
                        help="Use permutation importance to prune descriptors",
                        action="store_true")

    parser.add_argument("--all_descriptor", "-ad",
                        help="All descriptor is used (For test)",
                        action="store_true")

    # For black_box optimization
    parser.add_argument("--random_seed", "-rs",
                        help="Random seed for bayes",
                        type=int,
                        default=0)

    parser.add_argument("--n_descriptor", "-nd",
                        help="Number of descriptors using Gaussian process (pruned by random forest)",
                        type=int,
                        default=10)

    parser.add_argument("--n_probe", "-np",
                        help="Number of materials to set priority by one ML procedure (probably the number of jobs in the queue system would be sufficient)",
                        type=int,
                        default=10)

    parser.add_argument("--n_rand_basis", "-nrb",
                        help="Number of basis",
                        type=int,
                        default=0)

    parser.add_argument("--monitor", "-m",
                        help="Monitoring time (sec)",
                        type=int,
                        default=60)

    parser.add_argument("--restart_dir", "-rd",
                        help="Restart by using {restart_dir}/step_XX/result.json. Also can be specify step_XX dir.",
                        type=str)

    parser.add_argument("--blox", "-b",
                        help="Use blox",
                        action="store_true")

    parser.add_argument("--initial_priority", "-ip",
                        help="Specify step_*/result.json. Same initial priority will be set",
                        type=str)

    args = parser.parse_args()
    return args


