from . import diags
from .compare_groups import compare_groups
from .config import settings
from .metrics import *


def configure(**kwargs):
    settings.update(**kwargs)
