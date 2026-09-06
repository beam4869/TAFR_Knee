"""CHIM-distance post-hoc geometric comparator, explicitly not an NBI solver."""
import numpy as np


def chim_select(y,anchors):
    # Signed distance to anchor hyperplane, towards ideal. For singular CHIM,
    # use the minimum-norm least-squares normal and report its residual.
    normal=np.linalg.lstsq(anchors,np.ones(len(anchors)),rcond=None)[0]
    residual=float(np.linalg.norm(anchors@normal-1))
    depth=(1-y@normal)/np.linalg.norm(normal)
    return int(np.argmax(depth)),dict(chim_residual=residual,depth=float(depth.max()))
