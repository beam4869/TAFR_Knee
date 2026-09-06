"""Read the independent validation layer without confusing it with search data."""
import json
from experiments.common import RESULTS

def independent_records():
    path=RESULTS/'raw/validation_extensions.json'
    if not path.exists():raise FileNotFoundError('Run experiments.runners.extend_validation first')
    return json.loads(path.read_text())['results']

def reported_bound_failed(reported,validated,tolerance=1e-7):
    return None if reported is None or validated is None else bool(validated>reported+tolerance)
