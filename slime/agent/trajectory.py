"""Build a per-session training trajectory from multi-turn conversation data.

The :class:`TrajectoryManager` builds one trajectory per session. ``record_turn``
feeds in each turn (prompt messages + the served model's sglang snapshot),
routing it into a per-sid message tree; ``get_trajectory`` then linearizes that
tree into a ``list[Sample]`` of loss-masked training rows, tolerating TITO
re-tokenization drift via fork/replace.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import re as _re
import logging
import math
from collections.abc import Iterator
from typing import Any

import torch

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _canonical_message(msg: dict) -> tuple:
    """Normalize a history message for mount-point comparison, absorbing the
    client's lossless re-rendering so append-only turns don't false-fork.

    Claude Code re-serializes the whole history every turn (stateless API), so a
    replayed message rarely matches byte-for-byte what we recorded: (a) prompt
    cache_control breakpoints drift across turns, (b) a system/user content may
    flip between a plain string and a [{type:text}] block list, (c) injected
    <system-reminder> text (gitStatus/attribution/todo) changes per turn, (d)
    whitespace wobbles. None of these change what the model saw, so we strip
    them and compare the semantic residue. Real content divergence (different
    text or tool_calls) still differs and still forks. Comparison-only: the
    tree keeps each node's original message untouched."""
    m = copy.deepcopy(msg)

    def _strip_cc(o):
        if isinstance(o, dict):
            o.pop("cache_control", None)
            for v in o.values():
                _strip_cc(v)
        elif isinstance(o, list):
            for v in o:
                _strip_cc(v)
    _strip_cc(m)

    def _flat(c):
        if isinstance(c, list):
            out = []
            for x in c:
                if isinstance(x, dict) and x.get("type") == "text":
                    out.append(x.get("text", ""))
                else:
                    import json as _json
                    out.append(_json.dumps(x, ensure_ascii=False, sort_keys=True))
            return "".join(out)
        return c if isinstance(c, str) else ("" if c is None else str(c))

    txt = _flat(m.get("content"))
    txt = _re.sub(r"<system-reminder>.*?</system-reminder>", "", txt, flags=_re.S)
    txt = _re.sub(r"\s+", " ", txt).strip()

    tc = m.get("tool_calls")
    if tc:
        import json as _json
        tc = _json.dumps(tc, ensure_ascii=False, sort_keys=True)
    return (m.get("role"), txt, tc)

REPLAY_FIELDS = (
    "rollout_top_p_token_ids",
    "rollout_top_p_token_offsets",
    "rollout_top_p_log_probs",
    "rollout_topk_token_ids",
    "rollout_topk_log_probs",
    "rollout_routed_experts",
)


# ===========================================================================
# TurnRecord
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """One sglang ``/generate`` snapshot: the contract between an adapter and the
    manager. Adapters build it from a turn's prompt/output token ids; ``record_turn``
    consumes it."""

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)
    ill_formed: bool = False
    replay: dict[str, Any] | None = None
    weight_versions: list[str] = dataclasses.field(default_factory=list)
    spec_info: Sample.SpecInfo = dataclasses.field(default_factory=Sample.SpecInfo)
    prefix_cache_info: Sample.PrefixCacheInfo = dataclasses.field(default_factory=Sample.PrefixCacheInfo)


# ===========================================================================
# MessageNode
# ===========================================================================


class MessageNode:
    """One node in a session's routing tree, carrying a single chat message
    (``None`` for the dummy root and for an assistant leaf we generated but
    whose ``response_message`` was empty).

    The two kinds are distinguished by whether ``turn`` is set, which reflects
    WHERE the message came from:

    * **generated** (``turn is not None``): an assistant message the model
      actually generated this turn, fed in via ``record_turn``. ``turn`` holds
      its :class:`TurnRecord` -- the prompt/output ids, logprobs and finish
      reason that ``get_trajectory`` linearizes into training tokens.
    * **routing-only** (``turn is None``): the message came from the prompt, not
      from generation, so it only exists to route. This is every
      system/user/tool node, AND any assistant we did NOT generate: a foreign
      assistant the client replayed in a later prompt, or a prior generated turn
      demoted by the rewrite-merge in ``_try_merge_assistant_rewrite``.
    """

    def __init__(
        self,
        *,
        role: str | None = None,
        message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: MessageNode | None = None,
    ) -> None:
        self.role = role
        self.message = message
        self.metadata = dict(metadata or {})
        self.parent: MessageNode | None = parent
        self.children: list[MessageNode] = []
        self.turn: TurnRecord | None = None  # the generated TurnRecord, else None (routing-only)
        self.turn_index: int | None = None
        # Shared by sibling leaf paths; the first to reach it trains on it, the rest
        # re-emit it as loss_mask=0 context -- so each response is trained exactly once.
        self.response_trained: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: MessageNode) -> MessageNode:
        child.parent = self
        self.children.append(child)
        return child

    def path_from_root(self) -> list[MessageNode]:
        """Ordered list of nodes from the first non-root ancestor down to self."""
        chain: list[MessageNode] = []
        node: MessageNode | None = self
        while node is not None and not node.is_root:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator[MessageNode]:
        if not self.children:
            yield self
            return
        for c in self.children:
            yield from c.leaves()


# ===========================================================================
# drift classification — how an incoming turn's prompt relates to held tokens
# ===========================================================================


def _common_prefix_len(a: list[int], b: list[int], chunk: int = 4096) -> int:
    limit = min(len(a), len(b))
    matched = 0
    while matched < limit:
        chunk_end = min(matched + chunk, limit)
        if a[matched:chunk_end] == b[matched:chunk_end]:
            matched = chunk_end
        else:
            while matched < chunk_end and a[matched] == b[matched]:
                matched += 1
            return matched
    return matched


class DriftKind(enum.Enum):
    CLEAN = "clean"  # drift == 0: prompt_ids exactly extends held tokens; append the tail beyond them
    REALIGN = "realign"  # drift inside the most-recent response span and short incoming response; replace that span (loss_mask=0)
    REALIGN_TAIL = "realign_tail"  # drift falls entirely in a non-trained (loss_mask=0) tail (user/tool re-render); overwrite just that tail, no trained token touched -> never fork regardless of output length
    FORK = "fork"  # everything else: close this builder, open a fresh one as a fork


# ===========================================================================
# SampleBuilder — accumulates turns into one trainable Sample (fork closes it)
# ===========================================================================


class _SampleBuilder:
    """Accumulates a chain's turns into the token sequence of one ``Sample``.

    A chain of turns is appended one at a time via :meth:`append_turn`. Ideally
    each turn's prompt exactly extends the tokens we already hold, but a replayed
    turn rarely re-tokenizes byte-for-byte: TITO round-trips and chat-template
    re-rendering both perturb the ids of content we've already seen. The builder
    handles this drift in a source-agnostic way, classified by where and how far
    the prompt diverges from the held tokens (see :meth:`classify_token_drift`):

    * **CLEAN** -- no drift; append the prompt tail beyond what we hold.
    * **REALIGN** -- a short divergence inside the most-recent response span;
      overwrite that span from the prompt as loss_mask=0 and keep accumulating.
    * **FORK** -- divergence too large or too early to absorb; this builder is
      rejected and the caller closes it and opens a fresh one. That boundary is
      the "fork".

    Each surviving builder yields one Sample.
    """

    def __init__(self, fork_threshold: int) -> None:
        self._fork_threshold = fork_threshold
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.last_response_start_idx: int | None = None
        self.leading_prompt_len: int = 0
        self.replay: dict[str, Any] | None = None
        self.weight_versions: list[str] = []
        self.spec_info = Sample.SpecInfo()
        self.prefix_cache_info = Sample.PrefixCacheInfo()

    def classify_token_drift(self, turn: TurnRecord) -> DriftKind:
        """Decide how this builder should absorb ``turn``'s prompt.

        The incoming turn's prompt is expected to match the tokens this builder
        already holds as an exact prefix. When token drift has occurred -- the
        prompt diverges from the held tokens -- we decide whether to REALIGN
        (heal a short divergence inside the most-recent response span) or to FORK
        (``len(turn.output_ids) >= fork_threshold``, or the divergence sits too
        early to absorb). With no drift the turn is handled the CLEAN way -- a
        plain prefix extension.
        """
        # R3 includes prompt routes, which can differ even for identical token
        # prefixes. Token-local top-p / SC distributions can be concatenated,
        # but routing snapshots must remain separate model invocations.
        held_replay, new_replay = self.replay or {}, turn.replay or {}
        if (
            "rollout_routed_experts" in held_replay
            or "rollout_routed_experts" in new_replay
            or held_replay.keys() != new_replay.keys()
        ):
            logger.warning("DRIFT_FORK branch=replay_keys held=%s new=%s",
                           sorted(held_replay.keys()), sorted(new_replay.keys()))
            return DriftKind.FORK
        if "rollout_topk_token_ids" in held_replay and (
            torch.as_tensor(held_replay["rollout_topk_token_ids"]).shape[1:]
            != torch.as_tensor(new_replay["rollout_topk_token_ids"]).shape[1:]
        ):
            logger.warning("DRIFT_FORK branch=topk_shape")
            return DriftKind.FORK
        realign_at = _common_prefix_len(self.tokens, turn.prompt_ids)
        drift = len(self.tokens) - realign_at

        if drift == 0:
            return DriftKind.CLEAN
        if held_replay:
            logger.warning("DRIFT_FORK branch=held_replay drift=%d held_len=%d prompt_len=%d",
                           drift, len(self.tokens), len(turn.prompt_ids))
            return DriftKind.FORK

        # REALIGN only heals drift that falls inside the most-recent response span
        # (and is short); divergence anywhere earlier, or an empty builder, forks.
        start = self.last_response_start_idx
        if start is not None and realign_at >= start and len(turn.output_ids) < self._fork_threshold:
            return DriftKind.REALIGN
        # B/loss_mask-aware: if the divergence sits entirely in a non-trained tail
        # (every held token from realign_at onward has loss_mask=0, i.e. a user/tool
        # region re-rendered by the client), overwriting it loses no training signal.
        # Absorb it regardless of incoming output length -- this immunizes against
        # ANY user/tool drift (cache_control and whatever the SDK does next) without
        # enumerating drift types, and never touches trained assistant tokens.
        if start is not None and 0 < realign_at <= len(self.loss_mask) and not any(self.loss_mask[realign_at:]):
            logger.warning("REALIGN_TAIL drift=%d realign_at=%d held_len=%d prompt_len=%d out_len=%d (non-trained tail, absorb)",
                           drift, realign_at, len(self.tokens), len(turn.prompt_ids), len(turn.output_ids))
            return DriftKind.REALIGN_TAIL
        # 诊断:token级drift导致的FORK(re-tokenization)。打印drift位置/是否在最近response内/边界token
        _inside = (start is not None and realign_at >= start)
        _htok = self.tokens[realign_at] if realign_at < len(self.tokens) else None
        _ptok = turn.prompt_ids[realign_at] if realign_at < len(turn.prompt_ids) else None
        logger.warning(
            "DRIFT_FORK branch=token_drift drift=%d realign_at=%d held_len=%d prompt_len=%d "
            "inside_last_resp=%s out_ids_len=%d fork_threshold=%d held_tok=%s prompt_tok=%s",
            drift, realign_at, len(self.tokens), len(turn.prompt_ids),
            _inside, len(turn.output_ids), self._fork_threshold, _htok, _ptok,
        )
        return DriftKind.FORK

    def append_turn(self, turn: TurnRecord, kind: DriftKind, *, trained: bool = True) -> None:
        """Append one turn into this SampleBuilder, branching on ``kind``: for REALIGN
        we overwrite the already-saved response span, for CLEAN we just append this
        turn's prompt tail."""
        assert kind is not DriftKind.FORK, "append_turn called on a builder that would fork"

        is_first_turn = self.last_response_start_idx is None
        if turn.replay is not None:
            self._append_replay(turn.replay, gap=0 if is_first_turn else len(turn.prompt_ids) - len(self.tokens))

        # --- append this turn's prompt tail (loss_mask=0) ---
        if kind is DriftKind.REALIGN:
            self._align_to_prompt(turn.prompt_ids)  # drop the drifted tail, re-append from prompt
        elif kind is DriftKind.REALIGN_TAIL:
            # overwrite only the non-trained tail from the exact drift point, so
            # earlier tokens (including the most-recent trained assistant output)
            # stay byte-identical. Safe because that tail is all loss_mask=0.
            self._align_tail_from(_common_prefix_len(self.tokens, turn.prompt_ids), turn.prompt_ids)
        else:  # CLEAN: held tokens are an exact prefix of prompt_ids; append the tail beyond them
            self._append_tokens(turn.prompt_ids[len(self.tokens) :], loss_mask=0)

        # --- append this turn's generated response (loss_mask=1 unless re-emitted as context) ---
        self.last_response_start_idx = len(self.tokens)
        # Shared sampled prefixes keep their original probabilities alongside
        # replay metadata. Only the loss mask changes on subsequent branches.
        self._append_tokens(turn.output_ids, loss_mask=int(trained), logprobs=turn.output_log_probs)
        self.weight_versions.extend(turn.weight_versions)
        if trained:
            # Shared prefixes appear in multiple samples as masked context;
            # their generation counters belong to the first branch only.
            self.spec_info.spec_accept_token_num += turn.spec_info.spec_accept_token_num
            self.spec_info.spec_draft_token_num += turn.spec_info.spec_draft_token_num
            self.spec_info.spec_verify_ct += turn.spec_info.spec_verify_ct
            self.spec_info.completion_token_num += turn.spec_info.completion_token_num
            self.prefix_cache_info.cached_tokens += turn.prefix_cache_info.cached_tokens
            self.prefix_cache_info.total_prompt_tokens += turn.prefix_cache_info.total_prompt_tokens

        if is_first_turn:
            self.leading_prompt_len = len(turn.prompt_ids)

    def _append_replay(self, replay: dict[str, Any], *, gap: int) -> None:
        """Keep sampler distributions aligned across masked tool/context tokens."""
        incoming = {key: torch.as_tensor(value) for key, value in replay.items()}
        if self.replay is None:
            self.replay = incoming
            return
        assert gap >= 0 and self.replay.keys() == incoming.keys()
        assert "rollout_routed_experts" not in incoming
        if "rollout_top_p_token_offsets" in incoming:
            offsets = self.replay["rollout_top_p_token_offsets"]
            end = int(offsets[-1])
            self.replay["rollout_top_p_token_offsets"] = torch.cat(
                (offsets, offsets.new_full((gap,), end), incoming["rollout_top_p_token_offsets"][1:] + end)
            )
            for key in ("rollout_top_p_token_ids", "rollout_top_p_log_probs"):
                if key in incoming:
                    self.replay[key] = torch.cat((self.replay[key], incoming[key]))
        if "rollout_topk_token_ids" in incoming:
            k = incoming["rollout_topk_token_ids"].shape[-1]
            for key in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
                value = self.replay[key]
                padding = (
                    torch.arange(k, device=value.device, dtype=value.dtype).expand(gap, k)
                    if key.endswith("token_ids")
                    else value.new_full((gap, k), -math.log(k))
                )
                self.replay[key] = torch.cat((value, padding, incoming[key]))

    def _align_to_prompt(self, prompt_ids: list[int]) -> None:
        """Heal REALIGN drift by overwriting the most-recent response span with
        ``prompt_ids`` as loss_mask=0: the drifted tokens carry no signal, and re-appending
        from the prompt keeps the builder contiguous. Earlier turns are untouched."""
        response_start = self.last_response_start_idx
        tail = prompt_ids[response_start:]
        self.tokens[response_start:] = tail
        self.loss_mask[response_start:] = [0] * len(tail)
        self.logprobs[response_start:] = [0.0] * len(tail)

    def _align_tail_from(self, start_idx: int, prompt_ids: list[int]) -> None:
        """Overwrite held tokens from ``start_idx`` (the exact drift point) with the
        prompt's tail as loss_mask=0. Caller guarantees loss_mask[start_idx:] is all
        zero, so no trained token is disturbed and token/mask/logprob stay aligned."""
        tail = prompt_ids[start_idx:]
        self.tokens[start_idx:] = tail
        self.loss_mask[start_idx:] = [0] * len(tail)
        self.logprobs[start_idx:] = [0.0] * len(tail)

    def _append_tokens(self, ids: list[int], *, loss_mask: int, logprobs: list[float] | None = None) -> None:
        self.tokens.extend(ids)
        self.loss_mask.extend([loss_mask] * len(ids))
        self.logprobs.extend(logprobs if logprobs else [0.0] * len(ids))

    def has_trained_response(self) -> bool:
        return any(self.loss_mask[self.leading_prompt_len :])

    def to_sample(
        self, base_sample: Sample, extra_metadata: dict[str, Any] | None, max_sample_tokens: int = 0
    ) -> Sample:
        """Emit the accumulated tokens as one ``Sample``, stripping the first-turn
        prompt so loss_mask / logprobs cover only the response region."""
        start = self.leading_prompt_len  # first-turn prompt stripped; response region starts here
        tokens = list(self.tokens)
        loss_mask = self.loss_mask
        logprobs = self.logprobs
        if max_sample_tokens and len(tokens) > max_sample_tokens:
            tokens = tokens[:max_sample_tokens]
            loss_mask = loss_mask[:max_sample_tokens]
            logprobs = logprobs[:max_sample_tokens]
        md = dict(extra_metadata or {})
        sample = Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            tokens=tokens,
            response_length=len(loss_mask) - start,
            loss_mask=loss_mask[start:],
            rollout_log_probs=logprobs[start:],
            reward=0.0,
            status=Sample.Status.COMPLETED,
            metadata=md,
            weight_versions=list(self.weight_versions),
            spec_info=dataclasses.replace(self.spec_info),
            prefix_cache_info=dataclasses.replace(self.prefix_cache_info),
        )
        if getattr(base_sample, "trace", None) is not None:
            # Retain the carrier until the orchestrator closes cleanup spans.
            sample.trace = base_sample.trace
        if self.replay is not None:
            count = sample.response_length
            for key in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
                if key in self.replay:
                    setattr(sample, key, self.replay[key][:count])
            if "rollout_top_p_token_offsets" in self.replay:
                offsets = self.replay["rollout_top_p_token_offsets"][: count + 1]
                end = int(offsets[-1])
                sample.rollout_top_p_token_offsets = offsets
                sample.rollout_top_p_token_ids = self.replay["rollout_top_p_token_ids"][:end]
                if "rollout_top_p_log_probs" in self.replay:
                    sample.rollout_top_p_log_probs = self.replay["rollout_top_p_log_probs"][:end]
            if "rollout_routed_experts" in self.replay:
                sample.rollout_routed_experts = self.replay["rollout_routed_experts"][: max(0, len(tokens) - 1)]
        return sample


# ===========================================================================
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    def __init__(self, *, fork_threshold_tokens: int | None = None) -> None:
        self._fork_threshold: int = 1024 if fork_threshold_tokens is None else fork_threshold_tokens
        self._trees: dict[str, MessageNode] = {}
        self._turn_count: dict[str, int] = {}

    # -------------------- public ------------------------------------------

    def has_session(self, sid: str) -> bool:
        return sid in self._trees

    def turn_count(self, sid: str) -> int:
        return self._turn_count.get(sid, 0)

    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("record_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        assert not turn.output_log_probs or len(turn.output_log_probs) == len(turn.output_ids), (
            f"turn.output_log_probs length {len(turn.output_log_probs)} != "
            f"turn.output_ids length {len(turn.output_ids)}"
        )

        root = self._trees.setdefault(sid, MessageNode())

        node, depth = self._find_mount_point(root, prompt_messages)
        node, depth = self._try_merge_assistant_rewrite(sid, node, prompt_messages, depth)
        node = self._mount_prompt_messages(node, prompt_messages[depth:])
        self._attach_assistant_leaf(sid, node, turn=turn, response_message=response_message, metadata=metadata)
        _nleaves = sum(1 for _ in root.leaves())
        if _nleaves > 1:
            logger.warning("MOUNT_DIAG sid=%s after turn: tree now has %d leaves (multi-leaf=fork) mount_depth=%d msgs=%d",
                           sid, _nleaves, depth, len(prompt_messages))

    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample: Sample,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
        max_sample_tokens: int = 0,
    ) -> list[Sample]:
        """Linearize this sid's routing tree into slime ``Sample`` objects and
        consume the session.

        Each routing leaf yields one or more Samples; ``reward`` is assigned in
        full to every emitted Sample (not split across them), so each trained
        turn carries the trajectory's outcome reward. The sid is dropped
        afterwards, so a second call for the same sid returns ``[]``.
        """
        root = self._trees.get(sid)
        if root is None:
            return []

        samples: list[Sample] = []
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(
                self._chain_to_samples(
                    chain, base_sample=base_sample, extra_metadata=extra_metadata, max_sample_tokens=max_sample_tokens
                )
            )

        for s in samples:
            s.reward = reward

        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return samples

    def rebuild_main_agent_sample(
        self,
        sid: str,
        ordered_wire_ids: list[str],
        *,
        base_sample,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
        max_sample_tokens: int = 0,
    ) -> list:
        """Rebuild ONE clean main-agent Sample from an explicit tool_use-id order.

        The caller (rl_runner) locates the main agent's final request in
        litellm_requests.jsonl and extracts the ordered wire tool_use ids of its
        assistant turns. Those ids uniquely index the generating turns this
        session recorded (each turn carries its wire ids in message[
        '_wire_tool_ids']). Walking them in order and appending into a single
        builder yields the true linear trajectory, with api-retry / fork / probe
        / sub-agent turns excluded automatically (their ids never appear in the
        main agent's final history). This sidesteps the tree's mount/split
        heuristics entirely -- the id order IS the ground-truth path.

        Returns [] if no turn is found (caller falls back to get_trajectory).
        """
        root = self._trees.get(sid)
        if root is None or not ordered_wire_ids:
            return []
        # index: wire_tool_id -> assistant MessageNode(with .turn)
        id2node: dict[str, MessageNode] = {}
        def _walk(n):
            if n.role == "assistant" and getattr(n, "turn", None) is not None:
                for wid in (n.message or {}).get("_wire_tool_ids", []) or []:
                    id2node.setdefault(wid, n)
            for c in n.children:
                _walk(c)
        _walk(root)

        # pick nodes in the main-agent id order (dedup: one node may own several ids)
        seen_nodes: set[int] = set()
        ordered_nodes = []
        for wid in ordered_wire_ids:
            node = id2node.get(wid)
            if node is not None and id(node) not in seen_nodes:
                seen_nodes.add(id(node))
                ordered_nodes.append(node)
        if not ordered_nodes:
            logger.warning("REBUILD sid=%s ordered_nodes EMPTY: wire_ids=%d id2node=%d (no id matched tree)",
                           sid, len(ordered_wire_ids), len(id2node))
            return []
        logger.info("REBUILD sid=%s wire_ids=%d id2node=%d ordered_nodes=%d (matched %d/%d)",
                    sid, len(ordered_wire_ids), len(id2node), len(ordered_nodes),
                    sum(1 for w in ordered_wire_ids if w in id2node), len(ordered_wire_ids))

        builder = _SampleBuilder(self._fork_threshold)
        forks = 0
        for _ni, node in enumerate(ordered_nodes):
            turn = node.turn
            if turn is None:
                logger.warning("REBUILD sid=%s node#%d has turn=None, skip", sid, _ni)
                continue
            try:
                if not builder.tokens:
                    builder.append_turn(turn, DriftKind.CLEAN, trained=True)
                    continue
                kind = builder.classify_token_drift(turn)
                if kind is DriftKind.FORK:
                    forks += 1
                    builder.append_turn(turn, DriftKind.CLEAN, trained=True)
                else:
                    builder.append_turn(turn, kind, trained=True)
            except Exception as _e:
                _pl = len(getattr(turn, "prompt_ids", []) or [])
                _ol = len(getattr(turn, "output_ids", []) or [])
                logger.warning("REBUILD sid=%s FAIL at node#%d/%d prompt_ids=%d output_ids=%d builder_tokens=%d err=%r",
                               sid, _ni, len(ordered_nodes), _pl, _ol, len(builder.tokens), _e)
                raise
        if forks:
            logger.warning("REBUILD sid=%s unexpected FORK x%d on main-agent id path (forced CLEAN)", sid, forks)

        if not builder.has_trained_response():
            return []
        sample = builder.to_sample(base_sample, extra_metadata, max_sample_tokens)
        sample.reward = reward
        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return [sample]

    def drop_session(self, sid: str) -> None:
        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)

    # -------------------- internals ----------------------------------------

    def _find_mount_point(self, root: MessageNode, messages: list[dict[str, Any]]) -> tuple[MessageNode, int]:
        """Walk down the tree matching each message by role and dict equality (==),
        returning the deepest node that still matches and where to mount the rest."""
        node = root
        depth = 0
        while depth < len(messages):
            msg = messages[depth]
            next_child = None
            _role = msg.get("role")
            # LAYERED_ALIGN: assistant turns were generated by the server; the
            # client replays them with lossless but non-identical edits (SDK adds
            # tool-call default fields like replace_all=false, reorders keys).
            # Under append-only rollout the n-th replayed assistant always maps to
            # the n-th generated assistant, so align assistants BY POSITION (role
            # only) when the mount node has exactly one assistant child -- never
            # comparing their content, which immunizes against default-field
            # injection. With >1 assistant child the tree already branched and
            # position is ambiguous, so fall back to canonical comparison.
            # user/tool messages stay content-matched (canonical) since those are
            # client-owned and must distinguish genuinely different histories.
            if _role == "assistant":
                _asst_children = [c for c in node.children if c.role == "assistant"]
                if len(_asst_children) == 1:
                    next_child = _asst_children[0]
                else:
                    _msg_canon = _canonical_message(msg)
                    for child in _asst_children:
                        if _canonical_message(child.message) == _msg_canon:
                            next_child = child
                            break
            else:
                _msg_canon = _canonical_message(msg)
                for child in node.children:
                    if child.role == _role and _canonical_message(child.message) == _msg_canon:
                        next_child = child
                        break
            if next_child is None:
                # MOUNT_DIAG: 挂载在此中断。若此节点已有同role的child,就是即将multi-leaf fork。
                if depth < len(messages):
                    _sibs = [c for c in node.children if c.role == msg.get("role")]
                    if _sibs:
                        _sc = _canonical_message(_sibs[0].message)
                        _a = repr(_sc[1][:80]); _b = repr(_msg_canon[1][:80])
                        _firstdiff = -1
                        _sa, _sb = _sc[1], _msg_canon[1]
                        for _k in range(min(len(_sa), len(_sb))):
                            if _sa[_k] != _sb[_k]:
                                _firstdiff = _k; break
                        logger.warning(
                            "MOUNT_DIAG break@depth=%d role=%s existing_same_role_children=%d "
                            "=> WILL MULTI-LEAF FORK. canon_role_eq=%s canon_tc_eq=%s firstdiff@%s",
                            depth, msg.get("role"), len(_sibs),
                            _sc[0] == _msg_canon[0], _sc[2] == _msg_canon[2], _firstdiff,
                        )
                        if _firstdiff >= 0:
                            logger.warning("MOUNT_DIAG   existing_canon=...%s...", repr(_sa[max(0,_firstdiff-25):_firstdiff+30]))
                            logger.warning("MOUNT_DIAG   newmsg_canon =...%s...", repr(_sb[max(0,_firstdiff-25):_firstdiff+30]))
                        else:
                            logger.warning("MOUNT_DIAG   canon text equal but tc/role differ: existing_tc=%s new_tc=%s",
                                           repr(str(_sc[2])[:120]), repr(str(_msg_canon[2])[:120]))
                        # 原始(未canonical)差异,看是什么漂移
                        import json as _json
                        _rawa = _json.dumps(_sibs[0].message, ensure_ascii=False)
                        _rawb = _json.dumps(msg, ensure_ascii=False)
                        for _k in range(min(len(_rawa), len(_rawb))):
                            if _rawa[_k] != _rawb[_k]:
                                logger.warning("MOUNT_DIAG   RAW firstdiff@%d existing=%s", _k, repr(_rawa[max(0,_k-20):_k+40]))
                                logger.warning("MOUNT_DIAG   RAW                 new     =%s", repr(_rawb[max(0,_k-20):_k+40]))
                                break
                        else:
                            logger.warning("MOUNT_DIAG   RAW equal (len e=%d n=%d)", len(_rawa), len(_rawb))
                break
            node = next_child
            depth += 1
        return node, depth

    def _try_merge_assistant_rewrite(
        self,
        sid: str,
        node: MessageNode,
        prompt_messages: list[dict[str, Any]],
        depth: int,
    ) -> tuple[MessageNode, int]:
        """Merge a short assistant-rewrite onto its node instead of forking.

        A harness may replay a prior assistant message slightly re-rendered (e.g.
        whitespace) in a later prompt. It no longer matches the node we generated,
        so it would fork -- stranding the original generated turn as a dead-end
        leaf that still emits its own training Sample. Instead we overwrite that
        node's message in place and stop training its generated content (demote to
        routing-only), so only the live branch trains. This only applies below
        ``fork_threshold``: a long abandoned response carries enough real signal
        to fork and train standalone.

        Forking is always safe (a rewrite mounts as routing-only); this is purely
        a cleanup. So we merge only when the mount point has exactly one assistant
        child that is a leaf, generated (``turn`` set), and short (response <
        ``fork_threshold``), and fork otherwise, since absorbing destroys a
        generated TurnRecord irreversibly.
        """
        if self._fork_threshold <= 0:
            return node, depth  # feature off
        if depth >= len(prompt_messages) or prompt_messages[depth].get("role") != "assistant":
            return node, depth  # genuine non-assistant history fork -> leave it

        asst_children = [c for c in node.children if c.role == "assistant"]
        if len(asst_children) != 1:
            if len(asst_children) > 1:
                logger.warning(
                    "record_turn(sid=%s turn=%s): %d assistant children at mount "
                    "point; can't tell which the rewrite targets, so forking.",
                    sid,
                    self._turn_count.get(sid, 0) + 1,
                    len(asst_children),
                )
            return node, depth

        rewritten_node = asst_children[0]
        if (
            rewritten_node.children
            or rewritten_node.turn is None
            or len(rewritten_node.turn.output_ids) >= self._fork_threshold
        ):
            return node, depth

        rewritten_node.metadata["merged_rewrite"] = {
            "abandoned_turn_index": rewritten_node.turn_index,
            "abandoned_response_tokens": len(rewritten_node.turn.output_ids),
        }
        rewritten_node.turn = None
        rewritten_node.turn_index = None
        rewritten_node.message = prompt_messages[depth]
        return rewritten_node, depth + 1

    def _mount_prompt_messages(
        self,
        node: MessageNode,
        remaining_messages: list[dict[str, Any]],
    ) -> MessageNode:
        for m in remaining_messages:
            node = node.add_child(MessageNode(role=m.get("role"), message=m))
        return node

    def _attach_assistant_leaf(
        self,
        sid: str,
        node: MessageNode,
        *,
        turn: TurnRecord,
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        asst = MessageNode(
            role="assistant",
            message=response_message,
            metadata=dict(metadata or {}),
        )
        asst.turn = turn
        asst.turn_index = self._turn_count.get(sid, 0) + 1
        node.add_child(asst)
        self._turn_count[sid] = asst.turn_index

    def _split_chain_into_builders(self, chain: list[MessageNode]) -> list[_SampleBuilder]:
        """Pack the chain's generated turns into per-Sample token builders.

        Turns flow into the current builder until one can't extend it as an
        exact prefix (re-tokenization drift past what we can drop); that turn
        opens a new builder -- a fork. A generated turn shared by sibling leaves
        is trained only on the first leaf to claim it; later leaves re-emit it
        as loss_mask=0 context so the shared prefix isn't double-counted.
        """
        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]

        builders: list[_SampleBuilder] = []
        for asst_node in asst_nodes:
            trained = not asst_node.response_trained
            asst_node.response_trained = True

            if not builders or (kind := builders[-1].classify_token_drift(asst_node.turn)) is DriftKind.FORK:
                builders.append(_SampleBuilder(self._fork_threshold))
                builders[-1].append_turn(asst_node.turn, DriftKind.CLEAN, trained=trained)
            else:
                builders[-1].append_turn(asst_node.turn, kind, trained=trained)
        return builders

    def _chain_to_samples(
        self,
        chain: list[MessageNode],
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
        max_sample_tokens: int = 0,
    ) -> list[Sample]:

        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]
        truncated = bool(asst_nodes) and asst_nodes[-1].turn.finish_reason == "length"
        use_tool = any(bool((n.message or {}).get("tool_calls")) for n in asst_nodes)
        ill_formed = any(n.turn.ill_formed for n in asst_nodes)
        md = {
            **(extra_metadata or {}),
            "truncated": truncated,
            "use_tool": use_tool,
            "ill_formed": ill_formed,
        }
        return [
            builder.to_sample(base_sample, md, max_sample_tokens)
            for builder in self._split_chain_into_builders(chain)
            if builder.has_trained_response()
        ]


__all__ = [
    "TrajectoryManager",
    "TurnRecord",
]
