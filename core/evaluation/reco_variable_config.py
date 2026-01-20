"""Evaluator for comparing event reconstruction methods."""

import numpy as np
from typing import Union, Optional, List, Tuple
import matplotlib.pyplot as plt
import os
import timeit
import keras
from core.reconstruction import (
    EventReconstructorBase,
    GroundTruthReconstructor,
    KerasFFRecoBase,
    CompositeNeutrinoComponentReconstructor,
)
from core.base_classes import KerasMLWrapper
from .evaluator_base import (
    PlotConfig,
    BootstrapCalculator,
    BinningUtility,
    FeatureExtractor,
    AccuracyCalculator,
    SelectionAccuracyCalculator,
    NeutrinoDeviationCalculator,
)
from .plotting_utils import (
    AccuracyPlotter,
    ConfusionMatrixPlotter,
    ComplementarityPlotter,
    ResolutionPlotter,
    NeutrinoDeviationPlotter,
    SelectionAccuracyPlotter,
    DistributionPlotter,
)
from .physics_calculations import (
    TopReconstructor,
    ResolutionCalculator,
    lorentz_vector_from_PtEtaPhiE_array,
    c_hel,
    c_han,
)
from core.utils import compute_pt_from_lorentz_vector_array


reconstruction_variable_configs = {
    "top_mass": {
        "compute_func": lambda l, j, n: TopReconstructor.compute_top_masses(
            *TopReconstructor.compute_top_lorentz_vectors(l, j, n)
        ),
        "extract_func": lambda X: TopReconstructor.compute_top_masses(
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4]),
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4]),
        ),
        "label": r"$m(t)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $m(t)$ Resolution",
            "ylabel_deviation": r"Mean Relative $m(t)$ Deviation",
        },
    },
    "ttbar_mass": {
        "compute_func": lambda l, j, n: TopReconstructor.compute_ttbar_mass(
            *TopReconstructor.compute_top_lorentz_vectors(l, j, n)
        ),
        "extract_func": lambda X: TopReconstructor.compute_ttbar_mass(
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4]),
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4]),
        ),
        "label": r"$m(t\overline{t})$ [GeV]",
        "combine_tops": False,
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $m(t\overline{t})$ Resolution",
            "ylabel_deviation": r"Mean Relative $m(t\overline{t})$ Deviation",
        },
    },
    "c_han": {
        "compute_func": lambda l, j, n: c_han(
            *TopReconstructor.compute_top_lorentz_vectors(l, j, n),
            lorentz_vector_from_PtEtaPhiE_array(l[:, 0, :4]),
            lorentz_vector_from_PtEtaPhiE_array(l[:, 1, :4]),
        ),
        "extract_func": lambda X: c_han(
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4]),
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4]),
            lorentz_vector_from_PtEtaPhiE_array(X["lepton_truth"][:, 0, :4]),
            lorentz_vector_from_PtEtaPhiE_array(X["lepton_truth"][:, 1, :4]),
        ),
        "label": r"$c_{\text{han}}$",
        "combine_tops": False,
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$c_{\text{han}}$ Resolution",
            "ylabel_deviation": r"Mean $c_{\text{han}}$ Deviation",
        },
    },
    "c_hel": {
        "compute_func": lambda l, j, n: c_hel(
            *TopReconstructor.compute_top_lorentz_vectors(l, j, n),
            lorentz_vector_from_PtEtaPhiE_array(l[:, 0, :4]),
            lorentz_vector_from_PtEtaPhiE_array(l[:, 1, :4]),
        ),
        "extract_func": lambda X: c_hel(
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4]),
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4]),
            lorentz_vector_from_PtEtaPhiE_array(X["lepton_truth"][:, 0, :4]),
            lorentz_vector_from_PtEtaPhiE_array(X["lepton_truth"][:, 1, :4]),
        ),
        "label": r"$c_{\text{hel}}$",
        "combine_tops": False,
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$c_{\text{hel}}$ Resolution",
            "ylabel_deviation": r"Mean $c_{\text{hel}}$ Deviation",
        },
    },
    "neutrino_mag": {
        "compute_func": lambda l, j, n: (
            np.linalg.norm(n[:, 0, :], axis=-1),
            np.linalg.norm(n[:, 1, :], axis=-1),
        ),
        "extract_func": lambda X: (
            np.linalg.norm(X["neutrino_truth"][:, 0, :], axis=-1),
            np.linalg.norm(X["neutrino_truth"][:, 1, :], axis=-1),
        ),
        "label": r"$|\vec{p}(\nu)|$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $|\vec{p}(\nu)|$ Resolution",
            "ylabel_deviation": r"Mean Relative $|\vec{p}(\nu)|$ Deviation",
        },
    },
    "top_pt": {
        "compute_func": lambda l, j, n: (
            compute_pt_from_lorentz_vector_array(
                TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0]
            ),
            compute_pt_from_lorentz_vector_array(
                TopReconstructor.compute_top_lorentz_vectors(l, j, n)[1]
            ),
        ),
        "extract_func": lambda X: (
            compute_pt_from_lorentz_vector_array(
                lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4])
            ),
            compute_pt_from_lorentz_vector_array(
                lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4])
            ),
        ),
        "label": r"$p_{T}(t)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $p_{T}(t)$ Resolution",
            "ylabel_deviation": r"Mean Relative $p_{T}(t)$ Deviation",
        },
    },
    "nu_px": {
        "compute_func": lambda l, j, n: (n[:, 0, 0], n[:, 1, 0]),
        "extract_func": lambda X: (
            X["neutrino_truth"][:, 0, 0],
            X["neutrino_truth"][:, 1, 0],
        ),
        "label": r"$p_{x}(\nu)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{x}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{x}(\nu)$ Deviation [GeV]",
        },
    },
    "nu_py": {
        "compute_func": lambda l, j, n: (n[:, 0, 1], n[:, 1, 1]),
        "extract_func": lambda X: (
            X["neutrino_truth"][:, 0, 1],
            X["neutrino_truth"][:, 1, 1],
        ),
        "label": r"$p_{y}(\nu)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{y}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{y}(\nu)$ Deviation [GeV]",
        },
    },
    "nu_pz": {
        "compute_func": lambda l, j, n: (n[:, 0, 2], n[:, 1, 2]),
        "extract_func": lambda X: (
            X["neutrino_truth"][:, 0, 2],
            X["neutrino_truth"][:, 1, 2],
        ),
        "label": r"$p_{z}(\nu)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{z}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{z}(\nu)$ Deviation [GeV]",
        },
    },
    "top_px": {
        "compute_func": lambda l, j, n: (
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0][..., 0],
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[1][..., 0],
        ),
        "extract_func": lambda X: (
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4])[..., 0],
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4])[..., 0],
        ),
        "label": r"$p_{x}(t)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{x}(t)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{x}(t)$ Deviation [GeV]",
        },
    },
    "top_py": {
        "compute_func": lambda l, j, n: (
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0][..., 1],
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[1][..., 1],
        ),
        "extract_func": lambda X: (
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4])[..., 1],
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4])[..., 1],
        ),
        "label": r"$p_{y}(t)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{y}(t)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{y}(t)$ Deviation [GeV]",
        },
    },
    "top_pz": {
        "compute_func": lambda l, j, n: (
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0][..., 2],
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[1][..., 2],
        ),
        "extract_func": lambda X: (
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4])[..., 2],
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4])[..., 2],
        ),
        "label": r"$p_{z}(t)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{z}(t)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{z}(t)$ Deviation [GeV]",
        },
    },
    "top_energy": {
        "compute_func": lambda l, j, n: (
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[0][..., 3],
            TopReconstructor.compute_top_lorentz_vectors(l, j, n)[1][..., 3],
        ),
        "extract_func": lambda X: (
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4])[..., 3],
            lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4])[..., 3],
        ),
        "label": r"$E(t)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $E(t)$ Resolution",
            "ylabel_deviation": r"Mean Relative $E(t)$ Deviation",
        },
    },
    "parallel_component_nu": {
        "compute_func": lambda l, j, n: (np.sum(
            n[:,0, :] * l[:,0, :3], axis=-1
        ) / np.linalg.norm(l[:,0, :3], axis=-1), np.sum(
            n[:,1, :] * l[:,1, :3], axis=-1
        ) / np.linalg.norm(l[:,1, :3], axis=-1)),
        "extract_func": lambda X: (np.sum(
            X["neutrino_truth"][:,0, :] * X["lepton_truth"][:,0, :3], axis=-1
        ) / np.linalg.norm(X["lepton_truth"][:,0, :3], axis=-1), np.sum(
            X["neutrino_truth"][:,1, :] * X["lepton_truth"][:,1, :3], axis=-1
        ) / np.linalg.norm(X["lepton_truth"][:,1, :3], axis=-1)),
        "label": r"$p_{\parallel}(\nu)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{\parallel}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{\parallel}(\nu)$ Deviation [GeV]",
        },
    },
    "perpendicular_component_nu": {
        "compute_func": lambda l, j, n: (np.linalg.norm(
            n[:,0, :] - (np.sum(
                n[:,0, :] * l[:,0, :3], axis=-1, keepdims=True
            ) / np.linalg.norm(l[:,0, :3])**2) * l[:,0, :3], axis=-1
        ), np.linalg.norm(
            n[:,1, :] - (np.sum(
                n[:,1, :] * l[:,1, :3], axis=-1, keepdims=True
            ) / np.linalg.norm(l[:,1, :3])**2) * l[:,1, :3], axis=-1
        )),
        "extract_func": lambda X: (np.linalg.norm(
            X["neutrino_truth"][:,0, :] - (np.sum(
                X["neutrino_truth"][:,0, :] * X["lepton_truth"][:,0, :3], axis=-1, keepdims=True
            ) / np.linalg.norm(X["lepton_truth"][:,0, :3])**2) * X["lepton_truth"][:,0, :3], axis=-1
        ), np.linalg.norm(
            X["neutrino_truth"][:,1, :] - (np.sum(
                X["neutrino_truth"][:,1, :] * X["lepton_truth"][:,1, :3], axis=-1, keepdims=True
            ) / np.linalg.norm(X["lepton_truth"][:,1, :3])**2) * X["lepton_truth"][:,1, :3], axis=-1
        )),
        "label": r"$p_{\perp}(\nu)$ [GeV]",
        "combine_tops": True,
        "use_relative_deviation": False,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{\perp}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{\perp}(\nu)$ Deviation [GeV]",
        },
    },
}
