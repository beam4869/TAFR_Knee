"""Official SNEE objective registry; numerical solvers live in the adapter."""
from experiments.methods.snee_adapter import SneeProblem
PROBLEMS=('ZLT1','GRV1','VFM1','ZLT1q','GRV2','DAS1','DO2DK','DO2DKtight','VFM1constr')
def make_problem(name,dimension=None):
    if name not in PROBLEMS:raise ValueError(f'Unknown official SNEE problem: {name}')
    return SneeProblem(name,dimension)
