import importlib
import importlib.util

import pytest


def test_import_train_qm9_does_not_start_training():
    module = importlib.import_module("scripts.train_qm9")
    assert hasattr(module, "main")
    assert hasattr(module, "require_pyg")


def test_require_pyg_behavior():
    module = importlib.import_module("scripts.train_qm9")
    if importlib.util.find_spec("torch_geometric") is None:
        with pytest.raises(ImportError, match="QM9 training requires torch_geometric"):
            module.require_pyg()
    else:
        module.require_pyg()
