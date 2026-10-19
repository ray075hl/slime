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
    """Drop a whole group if ANY trajectory in it is empty/aborted.

    install_failed / aborted rollouts come back as placeholder ABORTED samples
    (response_length 0, loss_mask [0]) that still occupy a slot in the group.
    If they slip into training the group contributes a non-8-multiple of real
    samples, so the global count stops being a multiple of n_samples_per_prompt
    and Megatron dynamic-batch packing asserts (could not produce N mbs ...
    alignment threshold). Dropping the ENTIRE group removes exactly
    n_samples_per_prompt samples, which keeps the global count 8-aligned.
    Complete groups (all trajectories have a real trained response) are kept
    regardless of reward spread.
    """
    for s in samples:
        traj = s[-1] if isinstance(s, list) else s
        aborted = getattr(traj, "status", None)
        is_aborted = aborted is not None and str(getattr(aborted, "value", aborted)).lower() == "aborted"
        rlen = int(getattr(traj, "response_length", 0) or 0)
        lm = getattr(traj, "loss_mask", None)
        empty_mask = (not lm) or (sum(lm) == 0)
        if is_aborted or rlen == 0 or empty_mask:
            return DynamicFilterOutput(keep=False, reason="group_has_empty_or_aborted")
    return DynamicFilterOutput(keep=True)
