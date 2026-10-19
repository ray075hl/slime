import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_reward_nonzero_std_with_fallback", "keep_full_groups_only"]


def check_reward_nonzero_std(args, samples: list[Sample | list[Sample]], **kwargs):
    # Agent generators return one list of context-window segments per trajectory.
    # Count its terminal reward once, independent of the number of segments.
    rewards = [(sample[-1] if isinstance(sample, list) else sample).get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_reward_nonzero_std_with_fallback(args, samples: list[Sample], **kwargs):
    """Prefer non-zero-std groups without triggering another sampling round."""

    output = check_reward_nonzero_std(args, samples, **kwargs)
    output.keep_when_insufficient = True
    return output


def keep_full_groups_only(args, samples: list, **kwargs):
    """Drop ONLY incomplete groups (fewer trajectories than n_samples_per_prompt).

    Keeps every complete group regardless of reward spread (all-0, all-1, mixed
    are all kept). The sole purpose is to protect the train-side micro-batch
    alignment: when an instance has install_failed/aborted rollouts, its group
    ends up with <n_samples trajectories, so the global sample count stops being
    a multiple of n_samples_per_prompt and Megatron's dynamic-batch packing
    asserts (`could not produce N mbs ... alignment threshold`). Dropping those
    partial groups restores alignment without discarding any trainable signal.
    """
    n = int(getattr(args, "n_samples_per_prompt", 0) or 0)
    k = len(samples)
    keep = (n <= 0) or (k >= n)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"partial_group_{k}_of_{n}",
    )
