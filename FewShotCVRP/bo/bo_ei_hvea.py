import numpy as np
from scipy.optimize import minimize

import FewShotCVRP.bo.bo_pure as bo_pure
from hillvallimpl import HillVallEAConfiguration, optimize


class BayesianOptimizerEIHVEA(bo_pure.BayesianOptimizer):
    def _argmax_acq_multi(self, k: int) -> np.ndarray:

        def fun(x):
            ei = self._ei(x)
            return -ei  # minimize negative EI

        config = HillVallEAConfiguration(
            fun,
            self.bounds.shape[0],
            min(10**5, 5000 * self.bounds.shape[0]),
            10**9,
            internal_optimizer_name="CMSA-ES",
            lower=self.bounds[:, 0],
            upper=self.bounds[:, 1],
            random_seed=42,
            TargetTolFun=10.0,
        )
        starts = optimize(config)

        xs = []
        vals = []
        for x0 in starts:
            res = minimize(
                fun, x0=x0, method="L-BFGS-B", jac=False, bounds=self.bounds.tolist()
            )
            if res.success and np.isfinite(res.fun):
                xs.append(res.x)
                vals.append(-res.fun)  # EI value

        xs = np.asarray(xs, dtype=float)
        vals = np.asarray(vals, dtype=float)

        # Compute min separation
        if self.min_separation is not None:
            min_sep = float(self.min_separation)
        else:
            ranges = self.bounds[:, 1] - self.bounds[:, 0]
            avg_range = float(np.mean(ranges))
            min_sep = max(self.diversity_frac * avg_range, 1e-12)

        sel_idx = self._nms_select(xs, vals, k, min_sep)
        if not self.acq_opt_callback is None:
            self.acq_opt_callback(self, xs, starts, xs[sel_idx])
        return xs[sel_idx]
