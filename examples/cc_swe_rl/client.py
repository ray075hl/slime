"""Slime-side generate function that calls remote cc-run RL service.

Wire-up:
    --custom-generate-function-path examples.cc_swe_rl.client.generate

This is a thin HTTP client — all heavy lifting (Docker, eval, trajectory
capture) happens on the cc-run service side.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import socket
import time
from typing import Any

import aiohttp

from slime.utils.processing_utils import load_tokenizer
from slime.utils.misc import SingletonMeta
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

CC_RUN_URL = os.environ.get("CC_RUN_URL", "http://localhost:8080")
CC_RUN_TIMEOUT = int(os.environ.get("CC_RUN_TIMEOUT", "3600"))


def _keepalive_socket_factory(addr_info):
    """Build a TCP socket with keepalive so the slime->cc-run connection is
    never judged idle by the Tencent CLB (default 900s idle timeout), which
    otherwise RSTs long rollouts (agent runs minutes with no bytes on the wire,
    producing the exact 900s connection resets we observed, 40/40). Verified:
    without keepalive an idle CLB conn is RST at 900.1s; with KEEPIDLE=60 it
    survived 1300s."""
    family, type_, proto = addr_info[0], addr_info[1], addr_info[2]
    sock = socket.socket(family, type_, proto)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        if hasattr(socket, "TCP_KEEPINTVL"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60)
        if hasattr(socket, "TCP_KEEPCNT"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 10)
    except OSError:
        pass
    return sock
# The env cluster cannot reach the HPC-internal router IP that slime derives
# from args.sglang_router_ip. When router access is bridged via an EIP/DNAT
# (or a cloud-container TCP forwarder), set this to the externally-reachable
# address so the env's adapter dials that instead of the unroutable internal IP.
CC_RUN_SGLANG_EXTERNAL_URL = os.environ.get("CC_RUN_SGLANG_EXTERNAL_URL", "").strip()


class _State(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        self.sglang_url = (
            CC_RUN_SGLANG_EXTERNAL_URL
            or f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        )
        if CC_RUN_SGLANG_EXTERNAL_URL:
            logger.info(
                "[cc_swe_rl] sglang_url overridden to external %s (internal router %s:%s)",
                CC_RUN_SGLANG_EXTERNAL_URL, args.sglang_router_ip, args.sglang_router_port,
            )


async def generate(args, sample: Sample, sampling_params: dict[str, Any]):
    """Per-sample generate function that calls cc-run service."""
    state = _State(args)
    md = sample.metadata or {}
    instance_id = md.get("instance_id") or sample.label or "unknown"

    t0 = time.time()
    try:
        payload = {
            "instance_ids": [instance_id],
            "sglang_url": state.sglang_url,
            "sampling_params": sampling_params,
            "max_context_len": state.max_context_len,
            "tool_parser": state.tool_parser,
            "reasoning_parser": state.reasoning_parser,
        }
        logger.info(
            "[cc_swe_rl] -> /rollout instance=%s sglang=%s tool_parser=%s reasoning_parser=%s "
            "max_ctx=%s sampling=%s",
            instance_id, state.sglang_url, state.tool_parser, state.reasoning_parser,
            state.max_context_len, sampling_params,
        )

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(socket_factory=_keepalive_socket_factory)
        ) as session:
            async with session.post(
                f"{CC_RUN_URL}/rollout",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=CC_RUN_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("[cc_swe_rl] %s service HTTP %s: %s", instance_id, resp.status, error_text[:500])
                    return [_abort(sample, f"service_error:{resp.status}")]
                data = await resp.json()
        logger.info("[cc_swe_rl] <- /rollout instance=%s http=200 elapsed=%.1fs", instance_id, time.time() - t0)

        results = data.get("results", [])
        if not results:
            logger.error("[cc_swe_rl] %s empty results in response", instance_id)
            return [_abort(sample, "empty_response")]

        result = results[0]
        status = result.get("status", "error")
        raw_segments = result.get("segments", [])
        logger.info(
            "[cc_swe_rl] %s result: status=%s reward=%s is_solved=%s applied=%s segments=%d err=%s",
            instance_id, status, result.get("reward"), result.get("is_solved"),
            result.get("applied_cleanly"), len(raw_segments), (result.get("error_msg") or "")[:200],
        )
        if status in ("error", "aborted"):
            return [_abort(sample, f"{status}:{result.get('error_msg', '')}")]

        if not raw_segments:
            logger.warning("[cc_swe_rl] %s no segments (status=%s)", instance_id, status)
            return [_abort(sample, "no_segments")]

        reward = float(result.get("reward", 0.0))
        is_solved = result.get("is_solved", False)
        applied_cleanly = result.get("applied_cleanly", True)
        elapsed = time.time() - t0

        trajectory_metadata = {
            **(sample.metadata or {}),
            "instance_id": instance_id,
            "is_solved": is_solved,
            "applied_cleanly": applied_cleanly,
            "elapsed_sec": elapsed,
        }

        # Assemble one Sample per segment dict returned over HTTP. New slime
        # dropped TokenSegment / fan_out_sample_segments; we inline the same
        # mapping here. Reward is assigned IN FULL to every emitted Sample
        # (new-main semantics: each trained turn carries the outcome reward,
        # not reward/k as the old fan-out did). Sibling samples share
        # rollout_id so rollout-averaging reducers don't over-count.
        shared_rollout_id = getattr(sample, "index", None)
        k = len(raw_segments)
        fanned: list[Sample] = []
        for i, seg in enumerate(raw_segments):
            prompt_ids = list(seg["prompt_ids"])
            response_ids = list(seg["response_ids"])
            loss_mask = list(seg["loss_mask"])
            logp = list(seg.get("rollout_log_probs", []))
            if not response_ids or len(logp) != len(response_ids):
                continue
            sub = sample if i == 0 else copy.copy(sample)
            sub.tokens = prompt_ids + response_ids
            sub.response_length = len(response_ids)
            sub.loss_mask = loss_mask
            sub.rollout_log_probs = logp
            sub.response = state.tokenizer.decode(response_ids, skip_special_tokens=False)
            sub.reward = reward
            sub.status = Sample.Status.COMPLETED
            sub.rollout_id = shared_rollout_id
            sub.metadata = {**trajectory_metadata, "segment_idx": i, "num_segments": k}
            fanned.append(sub)

        if not fanned:
            return [_abort(sample, "fan_out_empty")]

        tot_resp = sum(s.response_length for s in fanned)
        tot_mask1 = sum(sum(s.loss_mask) for s in fanned)
        logger.info(
            "[cc_swe_rl] %s: reward=%.2f solved=%s elapsed=%.1fs segments=%d resp_tokens=%d trainable=%d",
            instance_id, reward, is_solved, elapsed, len(fanned), tot_resp, tot_mask1,
        )
        return fanned

    except asyncio.TimeoutError:
        logger.error("[cc_swe_rl] %s: timeout after %ds", instance_id, CC_RUN_TIMEOUT)
        return [_abort(sample, "client_timeout")]
    except Exception as e:
        logger.exception("[cc_swe_rl] %s: generate failed: %s", instance_id, e)
        return [_abort(sample, f"exception:{type(e).__name__}")]


def _abort(sample: Sample, reason: str) -> Sample:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.reward = 0.0
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("[cc_swe_rl] aborted: %s", reason)
    return sample
