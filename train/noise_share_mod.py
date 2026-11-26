"""noise_share_mod: a Flower client mod that adds each client's share of the DP noise before
SecAgg+ masks its update (distributed DP).

Use it *inside* Flower's ``secaggplus_mod`` (mods run outermost first)::

    mod = NoiseShareMod(C=1.0, sigma=4.84, n=20, run_seed=20000, clipping_range=8.0)
    app = ClientApp(client_fn=..., mods=[secaggplus_mod, mod])

so it sees the client's plaintext reply and returns the protected one, which ``secaggplus_mod``
then quantises and masks. For every TRAIN message it:

1. takes the arrays the client returned in its fit reply (any number of float arrays, treated as
   one flat vector: the client's *update*, e.g. a gradient or a weight difference),
2. clips that vector to L2 norm ``C`` and adds N(0, (sigma/sqrt(n))^2) per coordinate, by calling
   ``common.protect(update, 3, C, sigma, n, generator=g)``. The n noise shares add up to
   N(0, sigma^2) in the SecAgg+ sum. ``sigma`` None or 0 means clip only (no noise),
3. counts the protected values with |x| > ``clipping_range`` (values SecAgg+ quantisation will
   cut) and reports the count in the reply metrics under ``metrics_key``, so the server can
   log it and refuse the round,
4. writes the protected arrays back into the reply (same shapes and dtypes).

Noise is reproducible: the torch generator ``g`` is seeded with
``noise_seed(run_seed, round, partition_id)``, where the round is the message's ``group_id``
(Flower's default and SecAgg+ workflows set it to the server round) and the partition id is
``context.node_config[partition_key]``.

Only ``common`` (common.protect) is imported from the project, so the mod can be reused with
any Flower 1.45 ClientApp whose fit reply is in the legacy FitRes layout (a NumPyClient).
"""
from __future__ import annotations

from logging import INFO

import numpy as np
import torch
from flwr.app import Context, Message
from flwr.app.message_type import MessageType
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.compat.common import recorddict_compat as compat
from flwr.supercore import log

import common

METRICS_KEY = "clipping_range_cuts"


def noise_seed(run_seed: int, server_round: int, partition_id: int) -> int:
    """32-bit seed for one client's noise in one round, derived from (run_seed, round, partition id)."""
    return int(np.random.SeedSequence([run_seed, server_round, partition_id]).generate_state(1)[0])


class NoiseShareMod:
    """Client mod: clip the outgoing update to C and add the noise share sigma/sqrt(n).

    Parameters
    ----------
    C : float
        L2 clipping norm of the whole update (all returned arrays together).
    sigma : float or None
        Std of the *total* noise in the sum of n updates. Each client adds sigma/sqrt(n).
        None or 0 means clip only (SecAgg+ with no DP noise).
    n : int
        Number of clients whose updates are summed.
    run_seed : int
        Base seed. The per-round, per-client seed is noise_seed(run_seed, round, partition_id).
    clipping_range : float or None
        SecAgg+ clipping range. Values of the protected update with |x| > clipping_range are
        counted and reported under ``metrics_key``. None skips the count (reports 0).
    partition_key : str
        node_config key holding the client's integer partition id (default "partition-id",
        as set by Flower's simulation engine).
    metrics_key : str
        Name of the metric the count is reported under.
    """

    def __init__(self, C: float, sigma: float | None, n: int, run_seed: int,
                 clipping_range: float | None = None, *, partition_key: str = "partition-id",
                 metrics_key: str = METRICS_KEY) -> None:
        if C is None or C <= 0:
            raise ValueError("C must be > 0")
        if n < 1:
            raise ValueError("n must be >= 1")
        if sigma is not None and sigma < 0:
            raise ValueError("sigma must be >= 0 or None")
        self.C, self.sigma, self.n, self.run_seed = float(C), sigma, int(n), int(run_seed)
        self.clipping_range = clipping_range
        self.partition_key, self.metrics_key = partition_key, metrics_key

    def protect_arrays(self, arrays: list[np.ndarray], server_round: int,
                       partition_id: int) -> tuple[list[np.ndarray], int]:
        """Clip + noise share on a list of arrays (treated as one flat vector). Returns the
        protected arrays and the number of values with |x| > clipping_range."""
        flat = torch.from_numpy(np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays]))
        gen = torch.Generator().manual_seed(noise_seed(self.run_seed, server_round, partition_id))
        out = common.protect(flat, 3, self.C, self.sigma, self.n, generator=gen).numpy()
        cuts = 0 if self.clipping_range is None else int(np.count_nonzero(np.abs(out) > self.clipping_range))
        res, i = [], 0
        for a in arrays:
            res.append(out[i:i + a.size].reshape(a.shape).astype(a.dtype, copy=False))
            i += a.size
        return res, cuts

    def __call__(self, msg: Message, ctxt: Context, call_next) -> Message:
        if msg.metadata.message_type != MessageType.TRAIN:
            return call_next(msg, ctxt)
        out_msg = call_next(msg, ctxt)
        if out_msg.has_error():
            return out_msg

        server_round = int(msg.metadata.group_id)
        partition_id = int(ctxt.node_config[self.partition_key])
        fit_res = compat.recorddict_to_fitres(out_msg.content, keep_input=True)
        arrays, cuts = self.protect_arrays(parameters_to_ndarrays(fit_res.parameters), server_round, partition_id)
        fit_res.parameters = ndarrays_to_parameters(arrays)
        fit_res.metrics = {**(fit_res.metrics or {}), self.metrics_key: cuts}
        log(INFO, "noise_share_mod: round %s partition %s: clipped to C=%g, noise share std %g, "
                  "%d values beyond clipping_range", server_round, partition_id, self.C,
            (self.sigma or 0.0) / np.sqrt(self.n), cuts)
        out_msg.content = compat.fitres_to_recorddict(fit_res, keep_input=True)
        return out_msg
