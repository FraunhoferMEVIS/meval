# Settings must be changed before importing anything else from meval!
from meval.config import settings
settings.update(N_bootstrap=10, N_test_permut=10, max_N_student_bootstrap=10)

from demos.isic.demo import isic_demo
from demos.mimic.demo import mimic_demo

import os
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def working_directory(path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)

def test_isic_demo():
    with working_directory("demos/isic"):
        isic_demo()

def test_mimic_demo():
    with working_directory("demos/mimic"):
        mimic_demo()

if __name__ == "__main__":
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None

    test_isic_demo()
    test_mimic_demo()