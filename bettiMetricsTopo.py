from typing import List 
from monai.metrics import CumulativeIterationMetric
from monai.utils.enums import MetricReduction
import gudhi as gd
import torch
import numpy as np

class BettiErrorMetric(CumulativeIterationMetric):

    def __init__(self) -> None:
        super().__init__()
        self._buffers = None

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        betti_err = []
        for b in range(y_pred.shape[0]):
            for c in range(1, y_pred.shape[1]):
                betti_true = self._compute_betti(y_true[b, 0]==c)
                betti_pred = self._compute_betti(y_pred[b, c])
                diff_tmp = np.abs(np.array(betti_true[:-1]) - np.array(betti_pred[:-1]))
                betti_err.append(diff_tmp)
        # B,N,3 (batch, num_patches, betti numbers)
        return torch.tensor([betti_err])

    def _compute_betti(self, patch: np.array) -> List[int]:
        # Compute betti numbers
        cc = gd.CubicalComplex(top_dimensional_cells=1-patch)
        cc.compute_persistence()
        bnum = cc.persistent_betti_numbers(np.inf, -np.inf)
        return bnum

    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # This will be problematic with multiple classes, but it's easy to fix
        return torch.cat(self._buffers[0], dim=0)

class BettiErrorLocalMetric(CumulativeIterationMetric):

    def __init__(self) -> None:
        super().__init__()
        self._buffers = None
        self.window_size = 128

    def _compute_tensor(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        w = self.window_size

        betti_err = []
        for b in range(y_pred.shape[0]):
            for c in range(1, y_pred.shape[1]):
                y_t = y_true[b, 0]==c
                y_p = y_pred[b, c]
                comb = itertools.product(*[range(0, s, w) for s in y_t.shape])
                tmp_list = []
                for y, x in comb:
                    betti_true = self._compute_betti(y_t[y:y+w, x:x+w])
                    betti_pred = self._compute_betti(y_p[y:y+w, x:x+w])
                    diff_tmp = np.abs(np.array(betti_true[:-1]) - np.array(betti_pred[:-1]))
                    tmp_list.append(diff_tmp)
                betti_err.append( np.mean(tmp_list, axis=0) )
                #from IPython import embed; embed(); asd
        # B,N,3 (batch, num_patches, betti numbers)
        return torch.tensor([betti_err])

    def _compute_betti(self, patch: np.array) -> List[int]:
        # Compute betti numbers
        cc = gd.CubicalComplex(top_dimensional_cells=1-patch)
        cc.compute_persistence()
        bnum = cc.persistent_betti_numbers(np.inf, -np.inf)
        return bnum

    def aggregate(self, reduction: MetricReduction | str | None = None
                       ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # This will be problematic with multiple classes, but it's easy to fix
        return torch.cat(self._buffers[0], dim=0)
