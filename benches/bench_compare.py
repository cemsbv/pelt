import numpy as np
from scipy import stats
from pelt import predict
from ruptures.detection import Pelt

# Generate random signal data with 2 groups
signal = np.r_[stats.norm(-1,1).rvs(2000), stats.norm(2, 1).rvs(2000)]
signal_2d = np.array([signal]).transpose()

def pelt_l1():
    """ L1 on pelt"""
    predict(signal_2d, penalty=10, segment_cost_function="l1", sum_method="naive")

def ruptures_l1():
    """ L1 on ruptures"""
    Pelt(model='l1').fit_predict(signal, pen=10)


def pelt_l2():
    """ L2 on pelt"""
    predict(signal_2d, penalty=10, segment_cost_function="l2", sum_method="naive")

def ruptures_l2():
    """ L2 on ruptures"""
    Pelt(model='l2').fit_predict(signal, pen=10)

__benchmarks__ = [
    (pelt_l1, ruptures_l1, "ruptures L1 vs pelt L1"),
    (pelt_l2, ruptures_l2, "ruptures L2 vs pelt L2"),
]

