__version__ = '0.2.0'

from .download_tte import download_data
from .classify_data import classify_patents, classify_tasks
from .calculate_exposure import measure_exposure
from .generate_descriptions import generate_descriptions


def create_sample(*args, **kwargs):
    from .sample_data import create_sample as _fn
    return _fn(*args, **kwargs)


def label_sample(*args, **kwargs):
    from .label_data import label_sample as _fn
    return _fn(*args, **kwargs)


def train_model(*args, **kwargs):
    from .train_model import train_model as _fn
    return _fn(*args, **kwargs)


def validate_model(*args, **kwargs):
    from .validate_model import validate_model as _fn
    return _fn(*args, **kwargs)


__all__ = [
    'download_data',
    'classify_patents', 'classify_tasks',
    'measure_exposure',
    'generate_descriptions',
    'create_sample', 'label_sample',
    'train_model', 'validate_model',
]
