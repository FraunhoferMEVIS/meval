from .config import settings
from .compare_groups import compare_groups
from .metrics import *
from . import diags

def configure(**kwargs):
    settings.update(**kwargs)
