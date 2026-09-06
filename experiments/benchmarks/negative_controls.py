"""Controls distinguish preference-stable solutions from geometric bulges."""
import numpy as np


def linear_front(n=101):
    x = np.linspace(0, 1, n)
    return np.column_stack([x, 1-x])


def unsupported_front():
    # K is the local convex kink at x=.4 (slope -.5 to -.05), but every
    # interior point lies above the anchor chord. Thus weighted sums miss K.
    x = np.unique(np.r_[np.linspace(0, 1, 501), .4, .6])
    f2 = np.interp(x, [0, .4, .6, 1], [1, .8, .79, 0])
    return np.column_stack([x, f2]), np.array([.4, .8])
