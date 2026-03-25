#!/usr/bin/env python3
"""
Phantom RunPod Orchestrator — manage GPU pods from the command line.

Usage:
    python runpod/orchestrator.py start        # deploy a new pod → setup → start pipeline → update .env
    python runpod/orchestrator.py resume       # resume a stopped pod (uses RUNPOD_POD_ID from .env)
    python runpod/orchestrator.py stop         # pause pod (models + venv persist on volume)
    python runpod/orchestrator.py terminate    # delete pod (network volume survives)
    python runpod/orchestrator.py status       # show state + address
    python runpod/orchestrator.py gpus         # list GPUs with display names and API IDs
    python runpod/orchestrator.py datacenters  # list all RunPod datacenters

Two deploy modes (set RUNPOD_DEPLOY_MODE in .env):

  ssh (development):
    1. Deploy pod (devel base image — runtime tag doesn't exist)
    2. SSH via RunPod proxy ({podHostId}@ssh.runpod.io) → clone repo → startup.sh → pipeline via nohup
    3. Wait for pipeline on proxy URL → update .env with wss:// address
    Code changes: git pull on the pod. No image rebuild.

  docker (production):
    1. Deploy pod (custom image with everything baked in)
    2. Wait for pipeline on proxy URL (auto-started via Docker CMD)
    3. Update .env with wss:// address
    Code changes: rebuild and push the Docker image.

Networking:
  - WebSocket: always via RunPod proxy ({pod_id}-9000.proxy.runpod.net, wss://)
  - SSH: via RunPod proxy ({podHostId}@ssh.runpod.io, port 22)
  - Only port 9000/tcp is exposed on the pod (no 8888 — avoids JupyterLab init)

Key gotchas (see runpod/TROUBLESHOOTING.md for full details):
  - GPU names in .env are DISPLAY names; orchestrator resolves to API IDs via GraphQL
  - SSH username comes from GraphQL machine.podHostId (not machineId, not SDK get_pod())
  - runpod/pytorch runtime tag doesn't exist; devel is the only option (7.1 GB)
  - support_public_ip=True slows scheduling; only used in SSH mode
  - Never request both volume_in_gb and network_volume_id

Reads from .env in the repo root.
"""

import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    import runpod
except ImportError:
    print("ERROR: runpod not installed. Run: pip install runpod")
    sys.exit(1)

# Repo root is one level up from runpod/
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)

_POLL_INTERVAL = 3      # seconds between status polls
_PORT_TIMEOUT = 300      # seconds to wait for port assignment
_SSH_TIMEOUT = 120       # seconds to wait for SSH port to be reachable
_PIPELINE_TIMEOUT = 120  # seconds to wait for pipeline to bind port 9000

# Pod config
_VOLUME_MOUNT = "/workspace"
_POD_NAME = "phantom"
_GRAPHQL_URL = "https://api.runpod.io/graphql"

# Remote paths (on the pod, under /workspace network volume) — SSH mode only
_REMOTE_PHANTOM_DIR = "/workspace/Phantom"
_REMOTE_VENV_PYTHON = "/workspace/venv/bin/python"
_REMOTE_STARTUP = "{}/runpod/startup.sh".format(_REMOTE_PHANTOM_DIR)
_REMOTE_PIPELINE = "{}/pipeline.py".format(_REMOTE_PHANTOM_DIR)
_PIPELINE_LOG = "/workspace/phantom-pipeline.log"


def _get_deploy_mode() -> str:
    """Return 'ssh' or 'docker' from RUNPOD_DEPLOY_MODE env var."""
    mode = (os.getenv("RUNPOD_DEPLOY_MODE") or "ssh").strip().lower()
    if mode not in ("ssh", "docker"):
        print("ERROR: RUNPOD_DEPLOY_MODE must be 'ssh' or 'docker', got '{}'".format(mode))
        sys.exit(1)
    return mode


def _get_exposed_ports(mode: str) -> str:
    """Return exposed port string. Only expose 9000/tcp (pipeline WebSocket)."""
    return "9000/tcp"


# ── Env helpers ────────────────────────────────────────────────────────────────

def _update_env_key(key: str, value: str) -> None:
    """Rewrite a single key=value line in .env, appending if not present."""
    if not _ENV_PATH.exists():
        print("WARNING: .env not found at {}, skipping update".format(_ENV_PATH))
        return

    text = _ENV_PATH.read_text()
    pattern = r"^{}=.*$".format(re.escape(key))
    replacement = "{}={}".format(key, value)

    if re.search(pattern, text, re.MULTILINE):
        new_text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    else:
        new_text = text.rstrip() + "\n{}\n".format(replacement)

    _ENV_PATH.write_text(new_text)
    print("  Updated .env  {}={}".format(key, value))


# ── RunPod API helpers ─────────────────────────────────────────────────────────

def _get_pod_info(pod_id: str) -> dict:
    """Fetch full pod dict, or empty dict if not found."""
    return runpod.get_pod(pod_id) or {}


def _get_pod_status(pod_id: str) -> str:
    """Return the pod's current desiredStatus, or 'unknown'."""
    return _get_pod_info(pod_id).get("desiredStatus") or "unknown"


def _get_port_address(pod_id: str, private_port: int) -> Optional[str]:
    """Return 'ip:public_port' for a public-IP pod port, or None."""
    pod = _get_pod_info(pod_id)
    ports = (pod.get("runtime") or {}).get("ports") or []
    for port in ports:
        if port.get("privatePort") == private_port and port.get("isIpPublic"):
            return "{}:{}".format(port["ip"], port["publicPort"])
    return None


def _get_proxy_ws_url(pod_id: str) -> str:
    """Return RunPod proxy WebSocket URL for port 9000."""
    return "{}-9000.proxy.runpod.net".format(pod_id)


def _get_ssh_command(pod_id: str) -> Optional[str]:
    """
    Return SSH user@host string for RunPod's SSH proxy.
    Queries GraphQL for machine.podHostId which gives the full SSH username.
    Format: {podHostId}@ssh.runpod.io
    Returns None if not yet available.
    """
    api_key = os.getenv("RUNPOD_API_KEY", "")
    query = """
    query Pod($podId: String!) {
      pod(input: { podId: $podId }) {
        machine {
          podHostId
        }
      }
    }
    """
    try:
        resp = requests.post(
            _GRAPHQL_URL,
            json={"query": query, "variables": {"podId": pod_id}},
            headers={"Authorization": "Bearer {}".format(api_key)},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("errors"):
            print("  ERROR: SSH query returned errors: {}".format(data["errors"]))
            return None

        pod = (data.get("data") or {}).get("pod") or {}
        machine = pod.get("machine") or {}
        pod_host_id = machine.get("podHostId")
        if pod_host_id:
            return "{}@ssh.runpod.io".format(pod_host_id)
        return None
    except Exception as exc:
        print("  ERROR: Could not query SSH info: {}".format(exc))
        return None


def _wait_for_running(pod_id: str) -> None:
    """Poll until pod status is RUNNING."""
    print("Waiting for pod to start (up to {}s)...".format(_PORT_TIMEOUT))
    deadline = time.time() + _PORT_TIMEOUT
    last_status = ""

    while time.time() < deadline:
        status = _get_pod_status(pod_id)
        if status != last_status:
            elapsed = int(_PORT_TIMEOUT - (deadline - time.time()))
            print("  [{}s] Pod status: {}".format(elapsed, status))
            last_status = status

        if status == "RUNNING":
            return
        time.sleep(_POLL_INTERVAL)

    print("ERROR: pod not running after {}s. Last status: {}".format(_PORT_TIMEOUT, last_status))
    sys.exit(1)


def _wait_for_ports_ssh(pod_id: str) -> Tuple[str, str]:
    """
    Wait for pod to be RUNNING, then resolve SSH and WS addresses.
    SSH uses RunPod proxy: {pod_id}-{machine_id}@ssh.runpod.io
    WS uses RunPod proxy: {pod_id}-9000.proxy.runpod.net
    Returns (ssh_user_host, ws_address).
    """
    _wait_for_running(pod_id)

    ws_address = _get_proxy_ws_url(pod_id)
    print("  WS: {} (proxy)".format(ws_address))

    # SSH via RunPod proxy — needs machine ID from pod info
    print("Waiting for SSH proxy assignment...")
    deadline = time.time() + _PORT_TIMEOUT
    while time.time() < deadline:
        ssh_cmd = _get_ssh_command(pod_id)
        if ssh_cmd:
            print("  SSH: {}".format(ssh_cmd))
            return ssh_cmd, ws_address
        time.sleep(_POLL_INTERVAL)

    print("ERROR: SSH proxy not assigned after {}s.".format(_PORT_TIMEOUT))
    sys.exit(1)


def _wait_for_port_docker(pod_id: str) -> str:
    """
    Wait for pod to be RUNNING, then return WS proxy address.
    """
    _wait_for_running(pod_id)
    ws_address = _get_proxy_ws_url(pod_id)
    print("  WS: {} (proxy)".format(ws_address))
    return ws_address


def _get_datacenters(api_key: str) -> List[dict]:
    """
    Query RunPod GraphQL for all available datacenters.
    Returns list of dicts with 'id', 'name', 'location' keys.
    """
    query = """
    query {
      dataCenters {
        id
        name
        location
      }
    }
    """
    try:
        resp = requests.post(
            _GRAPHQL_URL,
            json={"query": query},
            headers={"Authorization": "Bearer {}".format(api_key)},
            timeout=10,
        )
        resp.raise_for_status()
        return (resp.json().get("data") or {}).get("dataCenters") or []
    except Exception as exc:
        print("ERROR: Could not query datacenters ({}).".format(exc))
        return []


def _get_gpu_types(api_key: str) -> List[dict]:
    """
    Query RunPod GraphQL for all GPU types with id and displayName.
    Returns list of dicts: [{"id": "...", "displayName": "..."}, ...].
    """
    query = """
    query {
      gpuTypes {
        id
        displayName
      }
    }
    """
    try:
        resp = requests.post(
            _GRAPHQL_URL,
            json={"query": query},
            headers={"Authorization": "Bearer {}".format(api_key)},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("errors"):
            print("  ERROR: gpuTypes query returned errors: {}".format(data["errors"]))
            return []

        return (data.get("data") or {}).get("gpuTypes") or []
    except Exception as exc:
        print("  ERROR: Could not query GPU types: {}".format(exc))
        return []


def _deploy_new_pod(mode: str) -> str:
    """
    Deploy a fresh pod, trying each GPU in RUNPOD_GPU_TYPES in priority order.
    Pre-filters to GPUs present in RUNPOD_DATACENTER_ID before attempting creation.
    Updates RUNPOD_POD_ID in .env. Returns the new pod ID.
    """
    gpu_types_raw = os.getenv("RUNPOD_GPU_TYPES", "")
    image = os.getenv("RUNPOD_IMAGE")
    datacenter_id = os.getenv("RUNPOD_DATACENTER_ID") or None
    container_disk = int(os.getenv("RUNPOD_CONTAINER_DISK", "20"))
    volume_disk = int(os.getenv("RUNPOD_VOLUME_DISK", "20"))
    network_volume_id = os.getenv("RUNPOD_NETWORK_VOLUME_ID") or None
    api_key = os.getenv("RUNPOD_API_KEY", "")

    gpu_types = [g.strip() for g in gpu_types_raw.split(",") if g.strip()]

    if not gpu_types:
        print("ERROR: RUNPOD_GPU_TYPES not set in .env")
        sys.exit(1)
    if not image:
        print("ERROR: RUNPOD_IMAGE not set in .env")
        sys.exit(1)
    if not datacenter_id:
        print("ERROR: RUNPOD_DATACENTER_ID not set in .env — must match network volume location")
        sys.exit(1)

    exposed_ports = _get_exposed_ports(mode)

    print("Deploying new pod in datacenter {} [{}]...".format(datacenter_id, mode))
    print("  Image:  {}".format(image))
    if network_volume_id:
        print("  Volume: {} → {}".format(network_volume_id, _VOLUME_MOUNT))

    # Resolve display names to GPU IDs (create_pod needs the ID, not display name)
    print("  Fetching GPU type IDs...")
    all_gpus = _get_gpu_types(api_key)
    name_to_id = {gpu["displayName"]: gpu["id"] for gpu in all_gpus if gpu.get("displayName") and gpu.get("id")}

    candidates = []
    for name in gpu_types:
        gpu_id = name_to_id.get(name)
        if gpu_id:
            candidates.append((name, gpu_id))
        else:
            print("  WARNING: '{}' not found in RunPod GPU list — skipping".format(name))

    if not candidates:
        print("ERROR: none of your preferred GPUs matched RunPod's GPU list.")
        print("  Your list:  {}".format(", ".join(gpu_types)))
        print("  Available:  {}".format(", ".join(sorted(name_to_id.keys()))))
        sys.exit(1)

    print("  Trying in order: {}".format(", ".join("{} ({})".format(n, i) for n, i in candidates)))

    for gpu_name, gpu_id in candidates:
        print("  Trying {} [{}]...".format(gpu_name, gpu_id), end=" ", flush=True)
        try:
            create_kwargs = dict(
                name=_POD_NAME,
                image_name=image,
                gpu_type_id=gpu_id,
                gpu_count=1,
                container_disk_in_gb=container_disk,
                volume_mount_path=_VOLUME_MOUNT,
                ports=exposed_ports,
                data_center_id=datacenter_id,
                support_public_ip=(mode == "ssh"),
                start_ssh=(mode == "ssh"),
            )
            if network_volume_id:
                create_kwargs["network_volume_id"] = network_volume_id
            else:
                create_kwargs["volume_in_gb"] = volume_disk
            pod = runpod.create_pod(**create_kwargs)
            new_pod_id = pod["id"]
            print("ok")
            print("  Created pod: {} ({})".format(new_pod_id, gpu_name))
            _update_env_key("RUNPOD_POD_ID", new_pod_id)
            return new_pod_id
        except Exception as exc:
            print("unavailable ({})".format(exc))

    print("ERROR: none of the GPUs in RUNPOD_GPU_TYPES have capacity in {} right now.".format(datacenter_id))
    sys.exit(1)


def _wait_for_pipeline(ws_address: str) -> None:
    """Poll the pipeline with a real WebSocket health check.

    ws_address is either 'host:port' (public IP) or a proxy hostname
    like '{pod_id}-9000.proxy.runpod.net' (no port suffix → use 443/wss).
    """
    if ":" in ws_address and ws_address.rsplit(":", 1)[1].isdigit():
        ws_url = "ws://{}/ws".format(ws_address)
    else:
        ws_url = "wss://{}/ws".format(ws_address)

    print("\nWaiting for pipeline to be ready at {} (up to {}s)...".format(
        ws_url, _PIPELINE_TIMEOUT))
    deadline = time.time() + _PIPELINE_TIMEOUT

    while time.time() < deadline:
        try:
            from websockets.sync.client import connect
            with connect(ws_url, open_timeout=5, close_timeout=2) as ws:
                ws.send(json.dumps({"action": "health"}))
                reply = json.loads(ws.recv(timeout=5))
                if reply.get("status") == "healthy":
                    print("  Pipeline is ready (healthy).")
                    return
                print("  Unexpected health response: {}".format(reply))
        except ImportError:
            # websockets not installed locally — fall back to TCP check
            print("  WARNING: websockets not installed, falling back to TCP check.")
            _wait_for_pipeline_tcp(ws_address)
            return
        except Exception as exc:
            print("  Not ready: {}".format(exc))
            time.sleep(_POLL_INTERVAL)

    print("ERROR: Pipeline not healthy after {}s.".format(_PIPELINE_TIMEOUT))
    if _get_deploy_mode() == "ssh":
        print("SSH into the pod and check: tail -f {}".format(_PIPELINE_LOG))
    else:
        print("Check pod logs on the RunPod dashboard.")
    sys.exit(1)


def _wait_for_pipeline_tcp(ws_address: str) -> None:
    """Fallback: TCP-only readiness check (no WebSocket handshake)."""
    if ":" in ws_address and ws_address.rsplit(":", 1)[1].isdigit():
        host, port_str = ws_address.rsplit(":", 1)
        port = int(port_str)
    else:
        host = ws_address
        port = 443

    deadline = time.time() + _PIPELINE_TIMEOUT
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            print("  Port reachable (TCP only — could not verify health).")
            return
        except (socket.error, OSError) as exc:
            print("  Connection attempt failed: {}".format(exc))
            time.sleep(_POLL_INTERVAL)

    print("ERROR: Pipeline port not reachable after {}s.".format(_PIPELINE_TIMEOUT))
    sys.exit(1)


# ── SSH helpers (development mode) ────────────────────────────────────────────

def _require_paramiko() -> "module":  # type: ignore[name-defined]
    """Lazy-import paramiko so Docker mode never needs it installed."""
    try:
        import paramiko
        return paramiko
    except ImportError:
        print("ERROR: paramiko not installed. Required for ssh mode.")
        print("  Run: pip install paramiko")
        sys.exit(1)


def _load_ssh_key(key_path: str) -> object:
    """Load an SSH private key, trying ed25519, RSA, and ECDSA formats."""
    paramiko = _require_paramiko()
    path = os.path.expanduser(key_path)
    for key_class in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return key_class.from_private_key_file(path)
        except paramiko.ssh_exception.SSHException as exc:
            print("  {} failed for {}: {}".format(key_class.__name__, path, exc))
            continue
    print("ERROR: Could not load SSH key from {}. Supported formats: ed25519, RSA, ECDSA.".format(path))
    sys.exit(1)


def _wait_for_ssh_tcp(host: str, port: int) -> None:
    """Wait until the SSH port accepts a raw TCP connection."""
    print("  Waiting for SSH to be reachable at {}:{}...".format(host, port))
    deadline = time.time() + _SSH_TIMEOUT
    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            print("  SSH is up.")
            return
        except (socket.error, OSError) as exc:
            print("  Connection attempt failed: {}".format(exc))
            time.sleep(_POLL_INTERVAL)
    print("ERROR: SSH not reachable after {}s.".format(_SSH_TIMEOUT))
    sys.exit(1)


_SENTINEL = "@@PHANTOM_EXIT@@"
_SSH_CMD_TIMEOUT = 1800  # seconds — max time for any single SSH command (pip install can take 10+ min)

# ANSI escape code pattern (CSI sequences, OSC sequences, simple escapes)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[()][A-Z0-9]|\x1b[>=<]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes that RunPod's PTY injects."""
    return _ANSI_RE.sub("", text)


def _drain_channel(channel: object, timeout: float = 2.0) -> str:
    """Read all pending data from a channel until nothing arrives for *timeout* seconds."""
    channel.settimeout(timeout)  # type: ignore[attr-defined]
    buf = b""
    while True:
        try:
            chunk = channel.recv(4096)  # type: ignore[attr-defined]
            if chunk:
                buf += chunk
            else:
                break
        except socket.timeout:
            break
    return buf.decode("utf-8", errors="replace")


def _open_shell(client: object) -> object:
    """Open an interactive shell and prepare it for scripted command execution.

    RunPod's SSH proxy drops commands sent via exec_command — only interactive
    shell sessions actually execute. We open a shell, drain the MOTD banner,
    then disable echo and prompt so only command output and our sentinel reach
    the reader.
    """
    channel = client.invoke_shell(term="dumb", width=512, height=50)  # type: ignore[attr-defined]

    # Give the shell time to initialize and send MOTD
    time.sleep(3)
    banner = _drain_channel(channel, timeout=2.0)
    for line in banner.splitlines():
        clean = _strip_ansi(line).rstrip()
        if clean:
            sys.stdout.write("  " + clean + "\n")
    sys.stdout.flush()

    # Disable echo and prompt so we get clean output from commands
    channel.sendall(b"export PS1='' PS2=''; stty -echo 2>/dev/null\n")  # type: ignore[attr-defined]
    time.sleep(0.5)
    _drain_channel(channel, timeout=1.0)  # discard any residual output

    return channel


def _shell_run(channel: object, command: str, label: str) -> None:
    """Send a command through an interactive shell session, stream output, check exit code.

    Appends a sentinel + exit-code echo after the command so we know when
    it finishes and whether it succeeded. Strips ANSI escape codes before
    checking for the sentinel.
    """
    wrapped = '({cmd}) 2>&1; echo "{sentinel}$?"\n'.format(cmd=command, sentinel=_SENTINEL)
    print("\n[{}] $ {}".format(label, command))
    channel.sendall(wrapped.encode("utf-8"))  # type: ignore[attr-defined]

    exit_code = None
    buf = ""
    deadline = time.time() + _SSH_CMD_TIMEOUT

    while time.time() < deadline:
        channel.settimeout(30.0)  # type: ignore[attr-defined]
        try:
            chunk = channel.recv(4096)  # type: ignore[attr-defined]
        except socket.timeout:
            # No data for 30s — print a keepalive so user knows we're waiting
            sys.stdout.write("  [still running...]\n")
            sys.stdout.flush()
            continue

        if not chunk:
            break

        text = chunk.decode("utf-8", errors="replace")
        buf += text

        # Process complete lines from the buffer
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            clean = _strip_ansi(line).rstrip()
            if _SENTINEL in clean:
                code_str = clean.split(_SENTINEL)[-1].strip()
                try:
                    exit_code = int(code_str)
                except ValueError as exc:
                    print("  WARNING: Could not parse exit code from '{}': {}".format(clean, exc))
                    exit_code = 1
                break
            if clean:
                sys.stdout.write("  " + clean + "\n")
                sys.stdout.flush()

        if exit_code is not None:
            break
    else:
        print("ERROR: '{}' timed out after {}s.".format(label, _SSH_CMD_TIMEOUT))
        sys.exit(1)

    # Check leftover buffer for sentinel (in case no trailing newline)
    if exit_code is None:
        clean = _strip_ansi(buf).rstrip()
        if _SENTINEL in clean:
            code_str = clean.split(_SENTINEL)[-1].strip()
            try:
                exit_code = int(code_str)
            except ValueError:
                exit_code = 1

    if exit_code is None:
        print("WARNING: '{}' finished without exit code — assuming success.".format(label))
        exit_code = 0

    if exit_code != 0:
        print("ERROR: '{}' failed (exit {}).".format(label, exit_code))
        sys.exit(1)


def _ssh_setup_and_start(ssh_address: str, key_path: str) -> None:
    """
    SSH into the pod, clone repo if missing, run startup.sh, start pipeline via nohup.

    ssh_address is '{pod_id}-{machine_id}@ssh.runpod.io' (RunPod SSH proxy).

    Uses invoke_shell() instead of exec_command() because RunPod's SSH proxy
    silently drops commands sent via exec_command — only interactive shell
    sessions actually execute on the pod.

    All steps are idempotent:
    - git clone only runs if /workspace/Phantom does not exist (first deploy)
    - startup.sh skips venv install if /workspace/venv already exists (subsequent runs)
    """
    paramiko = _require_paramiko()

    # Parse 'user@host' format from RunPod SSH proxy
    username, host = ssh_address.rsplit("@", 1)
    port = 22

    _wait_for_ssh_tcp(host, port)

    key = _load_ssh_key(key_path)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Retry SSH connection — RunPod proxy accepts TCP immediately but the
    # container may still be starting, causing "container is not running" errors.
    _SSH_CONNECT_RETRIES = 12  # 12 * 10s = 2 minutes
    for attempt in range(1, _SSH_CONNECT_RETRIES + 1):
        try:
            print("  Connecting via SSH as {}@{}... (attempt {}/{})".format(
                username, host, attempt, _SSH_CONNECT_RETRIES))
            client.connect(hostname=host, port=port, username=username, pkey=key, timeout=30)
            shell = _open_shell(client)
            # Quick smoke test — if the container isn't ready, this will fail
            _shell_run(shell, "echo ready", "container-check")
            break
        except Exception as exc:
            client.close()
            if attempt == _SSH_CONNECT_RETRIES:
                print("ERROR: Container not ready after {} attempts: {}".format(
                    _SSH_CONNECT_RETRIES, exc))
                sys.exit(1)
            print("  Container not ready: {} — retrying in 10s...".format(exc))
            time.sleep(10)
            # Re-create client for next attempt
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Clone repo if not present — only runs on first-ever deploy
        repo_url = os.getenv("RUNPOD_REPO_URL")
        if repo_url:
            _shell_run(
                shell,
                "[ -d {dir} ] && echo 'Repo already exists, skipping clone.' || git clone --progress {url} {dir}".format(
                    dir=_REMOTE_PHANTOM_DIR, url=repo_url
                ),
                "git-clone",
            )
        else:
            # Verify the repo exists — if not, give a clear error
            _shell_run(
                shell,
                "[ -d {dir} ] || {{ echo 'ERROR: {dir} not found. Set RUNPOD_REPO_URL in .env to auto-clone.'; exit 1; }}".format(
                    dir=_REMOTE_PHANTOM_DIR
                ),
                "repo-check",
            )

        # Run startup.sh — installs ffmpeg, creates venv on first run
        _shell_run(shell, "bash {}".format(_REMOTE_STARTUP), "startup")

        # Kill any leftover pipeline process from a previous run
        _shell_run(
            shell,
            "pkill -f 'python.*pipeline.py' 2>/dev/null || true",
            "kill-old-pipeline",
        )

        # Start pipeline with nohup (no tmux dependency, survives SSH disconnect)
        pipeline_cmd = (
            "nohup {python} {pipeline} --execution-provider cuda"
            " > {log} 2>&1 &"
        ).format(
            python=_REMOTE_VENV_PYTHON,
            pipeline=_REMOTE_PIPELINE,
            log=_PIPELINE_LOG,
        )
        _shell_run(shell, pipeline_cmd, "pipeline-start")
        print("\n  Pipeline started (log: {}).".format(_PIPELINE_LOG))
        print("  To view logs: ssh into pod and run: tail -f {}".format(_PIPELINE_LOG))
    finally:
        client.close()


# ── Commands ───────────────────────────────────────────────────────────────────

def _boot_pod(active_pod_id: str, mode: str) -> None:
    """Shared boot sequence: wait for ports → setup → wait for pipeline → update .env."""
    if mode == "ssh":
        ssh_address, ws_address = _wait_for_ports_ssh(active_pod_id)
        key_path = os.getenv("RUNPOD_SSH_KEY_PATH", "~/.ssh/id_ed25519")
        _ssh_setup_and_start(ssh_address, key_path)
    else:
        ws_address = _wait_for_port_docker(active_pod_id)

    _wait_for_pipeline(ws_address)

    # Proxy URLs use wss:// (port 443); direct IPs use ws://
    if "proxy.runpod.net" in ws_address:
        _update_env_key("PHANTOM_API_URL", "wss://{}/ws".format(ws_address))
    else:
        _update_env_key("PHANTOM_API_URL", "ws://{}/ws".format(ws_address))

    print("\nDone. Open the desktop:")
    print("  python desktop.py")


def cmd_start() -> None:
    """Deploy a fresh pod and boot it. Always creates a new pod."""
    mode = _get_deploy_mode()
    print("Deploy mode: {}".format(mode))
    active_pod_id = _deploy_new_pod(mode)
    _boot_pod(active_pod_id, mode)


def cmd_resume(pod_id: str) -> None:
    """Resume a stopped pod by its ID."""
    mode = _get_deploy_mode()
    print("Deploy mode: {}".format(mode))

    pod = runpod.get_pod(pod_id)
    if pod is None:
        print("ERROR: Pod {} not found.".format(pod_id))
        sys.exit(1)

    print("Resuming pod {}...".format(pod_id))
    try:
        runpod.resume_pod(pod_id, gpu_count=1)
    except Exception as exc:
        print("ERROR: Resume failed: {}".format(exc))
        sys.exit(1)

    _boot_pod(pod_id, mode)


def cmd_stop(pod_id: str) -> None:
    """Stop (pause) the pod. /workspace volume is preserved."""
    print("Stopping pod {}...".format(pod_id))
    runpod.stop_pod(pod_id)
    print("Pod stopped. /workspace volume preserved — models intact.")


def cmd_terminate(pod_id: str) -> None:
    """Permanently delete the pod (network volume survives)."""
    confirm = input(
        "Terminate pod {}? Container deleted, network volume survives. [y/N] ".format(pod_id)
    )
    if confirm.strip().lower() != "y":
        print("Aborted.")
        return
    print("Terminating pod {}...".format(pod_id))
    runpod.terminate_pod(pod_id)
    print("Pod terminated.")


def cmd_gpus() -> None:
    """List all RunPod GPUs with their IDs, highlighting preferred ones."""
    api_key = os.getenv("RUNPOD_API_KEY", "")
    gpu_types_raw = os.getenv("RUNPOD_GPU_TYPES", "")
    preferred = {g.strip() for g in gpu_types_raw.split(",") if g.strip()}

    print("Querying RunPod GPUs...\n")
    all_gpus = _get_gpu_types(api_key)

    if not all_gpus:
        print("No GPUs found (query may have failed).")
        return

    all_gpus.sort(key=lambda g: g.get("displayName", ""))

    matched = [g for g in all_gpus if g.get("displayName") in preferred]
    others = [g for g in all_gpus if g.get("displayName") not in preferred]

    if matched:
        print("Preferred (in RUNPOD_GPU_TYPES):")
        for gpu in matched:
            print("  * {}  [id: {}]".format(gpu.get("displayName"), gpu.get("id")))

    if others:
        if matched:
            print()
        print("Other available GPUs:")
        for gpu in others:
            print("    {}  [id: {}]".format(gpu.get("displayName"), gpu.get("id")))

    print("\nTotal: {} GPUs".format(len(all_gpus)))

    available_names = {g.get("displayName") for g in all_gpus}
    missing = preferred - available_names
    if missing:
        print("\nNot found in RunPod: {}".format(", ".join(sorted(missing))))


def cmd_datacenters() -> None:
    """List all RunPod datacenters with their IDs."""
    api_key = os.getenv("RUNPOD_API_KEY", "")
    current_dc = os.getenv("RUNPOD_DATACENTER_ID") or None

    print("Querying RunPod datacenters...\n")
    datacenters = _get_datacenters(api_key)

    if not datacenters:
        print("No datacenters found (query may have failed).")
        return

    datacenters.sort(key=lambda dc: dc.get("id", ""))

    for dc in datacenters:
        dc_id = dc.get("id", "?")
        name = dc.get("name", "")
        location = dc.get("location", "")
        marker = " <-- current" if dc_id == current_dc else ""
        label = "{} ({})".format(name, location) if location else name
        print("  {}  {}{}".format(dc_id, label, marker))

    print("\nTotal: {} datacenters".format(len(datacenters)))
    if current_dc:
        print("Current RUNPOD_DATACENTER_ID: {}".format(current_dc))
    else:
        print("RUNPOD_DATACENTER_ID is not set in .env")


def cmd_status(pod_id: str) -> None:
    """Show pod status, GPU, cost, and current WebSocket address."""
    pod = runpod.get_pod(pod_id)

    if not pod:
        print("Pod {} not found.".format(pod_id))
        return

    status = pod.get("desiredStatus", "unknown")
    name = pod.get("name", pod_id)
    gpu = (pod.get("machine") or {}).get("gpuDisplayName", "unknown")
    cost = pod.get("costPerHr") or 0.0
    uptime = pod.get("uptimeSeconds") or 0

    print("Pod:    {} ({})".format(name, pod_id))
    print("Status: {}".format(status))
    print("GPU:    {}".format(gpu))
    print("Cost:   ${:.4f}/hr".format(float(cost)))

    if uptime:
        print("Uptime: {}h {}m".format(uptime // 3600, (uptime % 3600) // 60))

    if status == "RUNNING":
        proxy_url = _get_proxy_ws_url(pod_id)
        print("URL:    wss://{}/ws".format(proxy_url))
    else:
        print("URL:    not available (pod status: {})".format(status))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse args and dispatch to the right command."""
    parser = argparse.ArgumentParser(
        description="Phantom RunPod Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  start        Deploy a new pod, setup, start pipeline, update .env
  resume       Resume a stopped pod (uses RUNPOD_POD_ID from .env)
  stop         Pause pod (preserves /workspace volume — models intact)
  terminate    Permanently delete pod (network volume survives)
  status       Show pod state, GPU, cost, and current WebSocket address
  gpus         List GPUs available in RUNPOD_DATACENTER_ID
  datacenters  List all RunPod datacenters and their IDs

Set RUNPOD_DEPLOY_MODE=ssh (development) or docker (production) in .env.
        """,
    )
    parser.add_argument("command", choices=["start", "resume", "stop", "terminate", "status", "gpus", "datacenters"])
    args = parser.parse_args()

    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set in .env")
        sys.exit(1)
    runpod.api_key = api_key

    pod_id = os.getenv("RUNPOD_POD_ID") or None

    if args.command == "start":
        if pod_id:
            print("WARNING: RUNPOD_POD_ID is set ({}).".format(pod_id))
            print("  'start' will deploy a NEW pod (the existing one is not affected).")
            print("  Did you mean 'resume'?")
            answer = input("\nProceed with new pod? [y/N] ").strip().lower()
            if answer != "y":
                print("Aborted. To resume the existing pod: python runpod/orchestrator.py resume")
                sys.exit(0)
        cmd_start()
    elif args.command == "datacenters":
        cmd_datacenters()
    elif args.command == "gpus":
        cmd_gpus()
    elif args.command in ("resume", "stop", "terminate", "status"):
        if not pod_id:
            print("ERROR: RUNPOD_POD_ID not set in .env")
            sys.exit(1)
        dispatch = {
            "resume": cmd_resume,
            "stop": cmd_stop,
            "terminate": cmd_terminate,
            "status": cmd_status,
        }
        dispatch[args.command](pod_id)  # type: ignore[operator]


if __name__ == "__main__":
    main()
