from . import EventReconstructorBase
import numpy as np
from core.DataLoader import DataConfig

class GroundTruthReconstructor(EventReconstructorBase):
    def __init__(
        self, config: DataConfig, use_nu_flows=False
    ):
        super().__init__(
            config=config,
            assignment_name="Ground truth",
            full_reco_name="True Assignment+ " + r"$\nu^2$-Flows" if use_nu_flows else r"True $\nu$",
            perform_regression=False, use_nu_flows=use_nu_flows
        )
        self.config = config

    def predict_indices(self, data_dict):
        return data_dict["assignment_labels"]

    def reconstruct_neutrinos(self, data_dict):
        return super().reconstruct_neutrinos(data_dict)


class FixedPrecisionReconstructor(EventReconstructorBase):
    def __init__(
        self, config: DataConfig, precision=0.1, name="fixed_precision_reconstructor"
    ):
        super().__init__(config=config, name=name)
        self.precision = precision

    def predict_indices(self, data_dict):
        true_labels = data_dict["assignment_labels"].copy()

        # Vectorized: choose which events to swap
        n_events = true_labels.shape[0]
        swap_mask = np.random.rand(n_events) > self.precision

        # Perform swap only for selected events
        if np.any(swap_mask):
            # Use fancy indexing to swap columns 0 and 1 where mask is True
            temp = true_labels[swap_mask, :, 0].copy()
            true_labels[swap_mask, :, 0] = true_labels[swap_mask, :, 1]
            true_labels[swap_mask, :, 1] = temp

        return true_labels

