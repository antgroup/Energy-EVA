"""
Templates and placeholders for the model adapters that live in *this* repo
(i.e. trained in-house, not third-party SOTA baselines).

The leaderboard in `README.md` includes one such entry — `EnergyTS_V3.0`. The
underlying architecture and weights are proprietary and intentionally not
shipped in this public skeleton; what we register here is a **placeholder**
that raises a clear, actionable error if someone tries to evaluate it without
providing their own implementation. This keeps the model name visible in
`registry`, lets validation scripts list it without crashing, and gives users
a single, obvious place to plug in their own adapter.

If you have your own model adapter (subclass of `ArchAdapter`), you can use
either `EnergyTSV3RegistrationPlaceholder` as a template, or extend the
`univar_model_name_and_path` table at the bottom — both paths show the
expected signature.
"""
import os
from functools import partial
from typing import List

import fev

from time_series_portal.evaluation_utils.unify_predict_methods import (
    univar_predict_with_model,
    covariates_predict_with_model,
)
from time_series_portal.evaluation_utils.persistence_util import PersistenceUtil
from time_series_portal.evaluation_utils.register import registry


# ---------------------------------------------------------------------------
# EnergyTS_V3.0 — placeholder registration.
# The model is referenced in README.md leaderboard for benchmark comparison.
# Users wanting to reproduce its scores must supply their own:
#   - ArchAdapter subclass (e.g. QwenAdapter-style backbone)
#   - Trained weight directory (under MODEL_STORAGE_PATH)
# Then either replace the body of `energy_ts_v3_0` below, or remove this
# block and wire the model through the template at the bottom of this file.
# ---------------------------------------------------------------------------
@registry.register("EnergyTS_V3.0")
def energy_ts_v3_0(
        _tasks: List[fev.Task],
        _persistence_helper: PersistenceUtil,
        **kwargs,
):
    raise NotImplementedError(
        "EnergyTS_V3.0 is a proprietary in-house model and is not bundled with "
        "this public skeleton. To evaluate it locally, supply your own "
        "ArchAdapter subclass and trained weights, then replace the body of "
        "`energy_ts_v3_0` in time_series_portal/evaluation_methods/adapter_methods.py "
        "(or use the `univar_model_name_and_path` template at the bottom of the file)."
    )


# ---------------------------------------------------------------------------
# Generic template for batch-registering in-house adapters.
# Each entry produces both `<name>` (univar) and `<name>_covariates`
# (with-covariates) registrations from a single line. Uncomment the import
# of your adapter class and add your models to `univar_model_name_and_path`.
# ---------------------------------------------------------------------------
# from Core.Models.arch_adapter import MyModelAdapter  # noqa: F401

MODEL_STORAGE_PATH = 'path_to_your_model_storage'
univar_model_name_and_path: List[tuple] = [
    # ('univar_my_model_128m', 'my_model_128m'),
]


def _to_register_univar_function(
        _model_path,
        _tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs
):
    device = kwargs.get('device', 'cuda:0')
    from Core.Models.arch_adapter import MyModelAdapter  # noqa: F401
    adapter = MyModelAdapter.load_model(_model_path).to(device)
    adapter.eval()
    return univar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs
    )


def _to_register_covariates_function(
        _model_path,
        _tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs
):
    device = kwargs.get('device', 'cuda:0')
    from Core.Models.arch_adapter import MyModelAdapter  # noqa: F401
    adapter = MyModelAdapter.load_model(_model_path).to(device)
    adapter.eval()
    return covariates_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs
    )


for m_model_name, m_model_path in univar_model_name_and_path:
    m_model_path = os.path.join(MODEL_STORAGE_PATH, m_model_path)
    if not os.path.exists(m_model_path):
        raise ValueError(f'{m_model_path} is not a valid model path')
    registry.register_function(m_model_name, partial(_to_register_univar_function, m_model_path))
    registry.register_function(m_model_name + '_covariates',
                               partial(_to_register_covariates_function, m_model_path, )
                               )
