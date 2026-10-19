#!/bin/bash
# Run this ON THE CLOUD-CONTAINER node (the one an EIP can be mapped to).
#
# Purpose: a plain TCP forwarder ("water pipe") that bridges the env cluster to
# the slime SGLang router living on the HPC cluster. The HPC router cannot be
# mapped by an EIP; the cloud container can. So:
#
#   env -> EIP:<ext port> -> cloud-container:LISTEN_PORT -(this forwarder)-> HPC router
#
# It does NOT parse HTTP, so large bodies / long-running (minutes) 256k rollouts
# pass through fine. Pair it with:
#   * slime training: --sglang-router-ip <HPC_ROUTER_IP> --sglang-router-port <HPC_ROUTER_PORT>
#     (pins the router to a known, fixed IP so this forward target is stable)
#   * slime training: export CC_RUN_SGLANG_EXTERNAL_URL=http://<EIP>:<ext port>
#     (so slime tells the env to dial the EIP, not the unroutable HPC IP)
#   * your EIP/DNAT:   EIP:<ext port> -> cloud-container:LISTEN_PORT
set -x

# ---- fill these in ----
HPC_ROUTER_IP="${HPC_ROUTER_IP:?export HPC_ROUTER_IP=<HPC launcher internal IP running the slime router>}"
HPC_ROUTER_PORT="${HPC_ROUTER_PORT:-4600}"     # must match slime --sglang-router-port
LISTEN_PORT="${LISTEN_PORT:-30001}"            # cloud-container port the EIP maps to
LOG="${LOG:-/tmp/forward_to_hpc_router.log}"

# ---- preflight: can this cloud container even reach the HPC router IP? ----
# (run BEFORE the router is up: ICMP/port may fail, but at least prove L3 route)
echo "== preflight: ping HPC router IP (L3 reachability) =="
ping -c 2 -W 2 "${HPC_ROUTER_IP}" || echo "WARN: ping failed (ICMP may be blocked; not fatal)"

echo "== preflight: is the HPC router already up? (optional) =="
curl -sS -m 5 "http://${HPC_ROUTER_IP}:${HPC_ROUTER_PORT}/health" \
  && echo "  -> HPC router reachable" \
  || echo "  -> HPC router not reachable yet (fine if training not started); forwarder will still listen"

# ---- pick a forwarder: prefer socat, fall back to nginx hint ----
if command -v socat >/dev/null 2>&1; then
  echo "== starting socat forwarder: 0.0.0.0:${LISTEN_PORT} -> ${HPC_ROUTER_IP}:${HPC_ROUTER_PORT} =="
  # fork: one child per connection; reuseaddr: fast restart.
  # No idle timeout set, so long rollouts are never cut off.
  exec socat -d -d \
    TCP-LISTEN:${LISTEN_PORT},fork,reuseaddr \
    TCP:${HPC_ROUTER_IP}:${HPC_ROUTER_PORT} \
    2>&1 | tee -a "${LOG}"
else
  cat <<EOF
socat not found. Install it:
    apt-get update && apt-get install -y socat     # debian/ubuntu
    yum install -y socat                           # centos/rhel

Or use nginx stream instead (more robust, tunable timeouts). Add to nginx.conf
at the SAME level as 'http {' (not inside it):

    stream {
        server {
            listen ${LISTEN_PORT};
            proxy_pass ${HPC_ROUTER_IP}:${HPC_ROUTER_PORT};
            proxy_connect_timeout 10s;
            proxy_timeout 1h;         # long 256k rollouts
        }
    }

then: nginx -t && nginx -s reload   (or start nginx)
EOF
  exit 1
fi
