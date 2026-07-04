"""Modern validation: Combinatorial Purged Cross-Validation (CPCV), purging,
embargo, and Probability of Backtest Overfitting (PBO), following Lopez de
Prado, *Advances in Financial Machine Learning* (2018), Ch. 7 & 11.

Why this matters institutionally: a single walk-forward backtest path massively
understates the true variance of a strategy's Sharpe ratio and is highly
susceptible to path-dependent overfitting. CPCV generates many
train/test splits from combinatorial groupings of time blocks, purges
training observations whose label horizon overlaps the test set (preventing
leakage from overlapping forward-return windows), and embargoes a short
window after each test fold (preventing serial-correlation leakage back into
training). This is the standard top-tier quant-fund validation protocol.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class CPCVFoldResult:
    """
    Combination Purged Cross-Validation Fold Result.

    Attributes:
        fold_id (int): Unique identifier for the fold.
        test_groups (tuple[int, ...]): Indices of the test groups used in this fold.
        n_train_obs (int): Number of training observations in this fold.
        n_test_obs (int): Number of testing observations in this fold.
        oos_sharpe (float): Out-of-sample Sharpe ratio for this fold.
        oos_mean_return (float): Out-of-sample mean return for this fold.
        oos_vol (float): Out-of-sample volatility for this fold.
    """

    fold_id: int
    test_groups: tuple[int, ...]
    n_train_obs: int
    n_test_obs: int
    oos_sharpe: float
    oos_mean_return: float
    oos_vol: float


def _purge_and_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_horizon: int,
    embargo_frac: float,
    n_obs: int,
) -> np.ndarray:
    """
    Remove training observations that (a) overlap the test set's label
    horizon (purging) or (b) fall within the embargo window immediately after
    the test set (embargo). Returns the filtered train index array.

    Args:
        train_idx (np.ndarray): Array of training indices.
        test_idx (np.ndarray): Array of testing indices.
        label_horizon (int): Forward-looking window (in observations) used to build the
            target (e.g. 5-day forward return) -- used for purging.
        embargo_frac (float): Fraction of n_obs to embargo after each test block.
        n_obs (int): Total number of time-ordered observations.

    Returns:
        np.ndarray: Filtered array of training indices after purging and embargoing.
    """
    test_start, test_end = test_idx.min(), test_idx.max()
    embargo = int(n_obs * embargo_frac)

    purge_lo = test_start - label_horizon
    purge_hi = test_end + label_horizon + embargo

    mask = (train_idx < purge_lo) | (train_idx > purge_hi)
    return train_idx[mask]


def combinatorial_purged_cv(
    n_obs: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    label_horizon: int = 5,
    embargo_frac: float = 0.01,
):
    """
    Generate CPCV (train_idx, test_idx) splits.

    Args:
        n_obs (int): Total number of time-ordered observations.
        n_groups (int): Number of contiguous time blocks to partition the sample into.
        n_test_groups (int): Number of blocks combined to form each test fold (>=2 gives
            the "combinatorial" property -- multiple disjoint test paths).
        label_horizon (int): Forward-looking window (in observations) used to build the
            target (e.g. 5-day forward return) -- used for purging.
        embargo_frac (float): Fraction of n_obs to embargo after each test block.

    Yields:
        tuple[np.ndarray, np.ndarray, tuple[int, ...]]: A tuple containing the training indices, testing indices, and the combination of test groups for each fold.

    Raises:
        ValueError: If n_groups is not greater than n_test_groups by at least 1 or if n_obs is less than n_groups.
        ValueError: If n_obs is less than 20, indicating insufficient data for meaningful CPCV.
    """
    if n_groups < n_test_groups + 1:
        raise ValueError("n_groups must exceed n_test_groups by at least 1")
    if n_obs < n_groups:
        raise ValueError("n_obs must be >= n_groups")

    bounds = np.linspace(0, n_obs, n_groups + 1).astype(int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]

    for combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in combo])
        train_groups = [g for g in range(n_groups) if g not in combo]
        train_idx = (
            np.concatenate([groups[g] for g in train_groups])
            if train_groups
            else np.array([], dtype=int)
        )
        train_idx = _purge_and_embargo(
            train_idx, test_idx, label_horizon, embargo_frac, n_obs
        )
        yield train_idx, test_idx, combo


def run_cpcv_backtest(
    returns_by_position: pd.Series,
    n_groups: int = 6,
    n_test_groups: int = 2,
    label_horizon: int = 5,
    embargo_frac: float = 0.01,
) -> pd.DataFrame:
    """
    Apply Combinatorial Purged Cross-Validation (CPCV) to an already-realized daily strategy-return series.
    P&L is assumed to be computed at the position level by `backtest.run_backtest`. This function evaluates the path robustness of the realized strategy, answering how variable the Sharpe ratio is across many disjoint, purged/embargoed out-of-sample sub-paths of the same strategy. It is particularly useful when the signal itself has no fitted parameters that need per-fold re-estimation.

    Args:
        returns_by_position (pd.Series): A pandas Series containing the realized daily returns of the strategy, indexed by date.
        n_groups (int): Number of contiguous time blocks to partition the sample into.
        n_test_groups (int): Number of blocks combined to form each test fold (>=2 gives the "combinatorial" property -- multiple disjoint test paths).
        label_horizon (int): Forward-looking window (in observations) used to build the target (e.g. 5-day forward return) -- used for purging.
        embargo_frac (float): Fraction of n_obs to embargo after each test block.

    Returns:
        pd.DataFrame: A DataFrame containing the results of each CPCV fold, including fold ID, test groups, number of training and testing observations, out-of-sample Sharpe ratio, mean return, and volatility.
    """
    n_obs = len(returns_by_position)
    if n_obs < 20:
        raise ValueError("Need at least 20 observations to run CPCV meaningfully")
    records = []
    values = returns_by_position.values
    dates = returns_by_position.index

    for fold_id, (train_idx, test_idx, combo) in enumerate(
        combinatorial_purged_cv(
            n_obs, n_groups, n_test_groups, label_horizon, embargo_frac
        )
    ):
        if len(test_idx) < 2:
            continue
        oos = values[test_idx]
        mean_r = float(np.mean(oos))
        vol_r = float(np.std(oos, ddof=0))
        sharpe = float((mean_r / vol_r) * np.sqrt(252)) if vol_r > 0 else 0.0
        records.append(
            CPCVFoldResult(
                fold_id=fold_id,
                test_groups=combo,
                n_train_obs=len(train_idx),
                n_test_obs=len(test_idx),
                oos_sharpe=sharpe,
                oos_mean_return=mean_r,
                oos_vol=vol_r,
            )
        )
    df = pd.DataFrame([r.__dict__ for r in records])
    return df


def probability_of_backtest_overfitting(
    fold_sharpes: pd.Series, n_trials_simulated: int = 500, seed: int = 7
) -> tuple[float, np.ndarray]:
    """
    Estimate the probability of backtest overfitting using a simplified CSCV-style approach.

    Simply repeatedly split the fold Sharpe ratios into an in-sample "selection" half and an out-of-sample "evaluation" half; measure how often the best-of-IS choice underperforms the OOS median (logit of relative rank < 0 => overfit signature). Returns (PBO estimate, array of rank logits) for plotting.

    Args:
        fold_sharpes (pd.Series): A pandas Series containing the Sharpe ratios from each CPCV fold.
        n_trials_simulated (int): Number of simulations to run for the PBO estimate.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[float, np.ndarray]: A tuple containing the PBO estimate and an array of rank logits for plotting.
    """
    rng = np.random.default_rng(seed)
    sharpes = fold_sharpes.dropna().values
    n = len(sharpes)
    if n < 4:
        return float("nan"), np.array([])

    logits = []
    for _ in range(n_trials_simulated):
        idx = rng.permutation(n)
        half = n // 2
        is_idx, oos_idx = idx[:half], idx[half:]
        if len(is_idx) == 0 or len(oos_idx) == 0:
            continue
        best_is = np.argmax(sharpes[is_idx])
        best_is_value = sharpes[is_idx][best_is]
        oos_rank = (sharpes[oos_idx] < best_is_value).mean()
        oos_rank = min(max(oos_rank, 1e-6), 1 - 1e-6)
        logit = np.log(oos_rank / (1 - oos_rank))
        logits.append(logit)

    logits = np.array(logits)
    pbo = float((logits < 0).mean()) if len(logits) else float("nan")
    return pbo, logits


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): probability that the observed Sharpe ratio is genuinely positive after correcting for the multiple-testing bias of having tried `n_trials` candidate signal variants. Returns a probability in [0, 1] (via the normal CDF); higher = more confidence the Sharpe is not a multiple-testing artifact.

    Args:
        observed_sharpe (float): The observed Sharpe ratio of the strategy.
        n_trials (int): The number of independent trials (candidate signal variants) tested.
        n_obs (int): The number of observations used to compute the Sharpe ratio.
        skew (float): The skewness of the return distribution. Default is 0.0.
        kurtosis (float): The kurtosis of the return distribution. Default is 3.0.
        benchmark_sharpe (float): The benchmark Sharpe ratio to compare against. Default is 0.0.

    Returns:
        float: The deflated Sharpe ratio, representing the probability that the observed Sharpe ratio is genuinely positive after correcting for multiple-testing bias. A value closer to 1 indicates higher confidence that the Sharpe is not a multiple-testing artifact.
    """
    from scipy.stats import norm

    if n_obs <= 1:
        return float("nan")
    # expected max Sharpe under n_trials independent trials (Bailey-Lopez de Prado approx)
    euler_gamma = 0.5772156649
    if n_trials > 1:
        expected_max_sr = (1 - euler_gamma) * norm.ppf(
            1 - 1.0 / n_trials
        ) + euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
        expected_max_sr /= np.sqrt(n_obs)
    else:
        expected_max_sr = 0.0

    sr_std = np.sqrt(
        (1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2)
        / (n_obs - 1)
    )
    if sr_std <= 0:
        return float("nan")
    z = (observed_sharpe - benchmark_sharpe - expected_max_sr) / sr_std
    return float(norm.cdf(z))
