#!/usr/bin/env python3
"""
Phantom RunPod Orchestrator — manage GPU pods from the command line.

Usage:
    python runpod/orchestrator.py start      # resume or deploy pod → setup → start pipeline → update .env
    python runpod/orchestrator.py stop       # pause pod (models + venv persist on volume)
    python runpod/orchestrator.py terminate  # delete pod (network volume survives)
    python runpod/orchestrator.py status     # show state + address

Two deploy modes (set RUNPOD_DEPLOY_MODE in .env):

  ssh (development):
    1. Resume or deploy pod (generic base image)
    2. SSH in → clone repo → run startup.sh → start pipeline in tmux
    3. Wait for port 9000 → update .env
    Code changes: git pull on the pod. No image rebuild.

  docker (production):
    1. Resume or deploy pod (custom image with everything baked in)
    2. Wait for port 9000 (pipeline auto-starts via Docker CMD)
    3. Update .env
    Code changes: rebuild and push the Docker image.

Reads from .env in the repo root.
"""

import argparse
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
_PORT_TIMEOUT = 120      # seconds to wait for port assignment
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
_TMUX_SESSION = "phantom"


def _get_deploy_mode() -> str:
    """Return 'ssh' or 'docker' from RUNPOD_DEPLOY_MODE env var."""
    mode = (os.getenv("RUNPOD_DEPLOY_MODE") or "ssh").strip().lower()
    if mode not in ("ssh", "docker"):
        print("ERROR: RUNPOD_DEPLOY_MODE must be 'ssh' or 'docker', got '{}'".format(mode))
        sys.exit(1)
    return mode


def _get_exposed_ports(mode: str) -> str:
    """SSH mode needs port 22 exposed; Docker mode only needs 9000."""
    if mode == "ssh":
        return "8888/http,9000/tcp"
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

def _get_port_address(pod_id: str, private_port: int) -> Optional[str]:
    """Return 'ip:public_port' for an exposed pod port, or None if not ready."""
    pod = runpod.get_pod(pod_id)
    ports = ((pod or {}).get("runtime") or {}).get("ports") or []
    for port in ports:
        if port.get("privatePort") == private_port and port.get("isIpPublic"):
            return "{}:{}".format(port["ip"], port["publicPort"])
    return None


def _wait_for_ports_ssh(pod_id: str) -> Tuple[str, str]:
    """
    Poll until both SSH (22) and WebSocket (9000) ports are assigned by RunPod.
    Returns (ssh_address, ws_address) as 'ip:port' strings.
    """
    print("Waiting for pod ports to be assigned...")
    deadline = time.time() + _PORT_TIMEOUT

    while time.time() < deadline:
        ssh_addr = _get_port_address(pod_id, 22)
        ws_addr = _get_port_address(pod_id, 9000)
        if ssh_addr and ws_addr:
            print("  SSH: {}  WS: {}".format(ssh_addr, ws_addr))
            return ssh_addr, ws_addr
        time.sleep(_POLL_INTERVAL)

    print("ERROR: pod ports not assigned after {}s.".format(_PORT_TIMEOUT))
    sys.exit(1)


def _wait_for_port_docker(pod_id: str) -> str:
    """
    Poll until WebSocket (9000) port is assigned by RunPod.
    Returns ws_address as 'ip:port' string.
    """
    print("Waiting for pod port to be assigned...")
    deadline = time.time() + _PORT_TIMEOUT

    while time.time() < deadline:
        ws_addr = _get_port_address(pod_id, 9000)
        if ws_addr:
            print("  WS: {}".format(ws_addr))
            return ws_addr
        time.sleep(_POLL_INTERVAL)

    print("ERROR: pod port not assigned after {}s.".format(_PORT_TIMEOUT))
    sys.exit(1)


def _get_gpus_in_datacenter(datacenter_id: str, api_key: str) -> List[str]:
    """
    Query RunPod GraphQL for GPU displayNames available in the given datacenter.
    Returns empty list on failure — callers fall back to trying all listed GPUs.
    """
    query = """
    query {
      gpuTypes {
        displayName
        nodeGroupDatacenters {
          id
        }
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
        gpu_types = (resp.json().get("data") or {}).get("gpuTypes") or []
        return [
            gpu["displayName"]
            for gpu in gpu_types
            if datacenter_id in [dc["id"] for dc in (gpu.get("nodeGroupDatacenters") or [])]
        ]
    except Exception as exc:
        print("  WARNING: Could not query GPU availability ({}). Trying all listed GPUs.".format(exc))
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

    print("  Checking which preferred GPUs are available in {}...".format(datacenter_id))
    available_in_dc = _get_gpus_in_datacenter(datacenter_id, api_key)

    if available_in_dc:
        candidates = [g for g in gpu_types if g in available_in_dc]
        if not candidates:
            print("ERROR: none of your preferred GPUs are offered in {}.".format(datacenter_id))
            print("  Your list:  {}".format(", ".join(gpu_types)))
            print("  Available:  {}".format(", ".join(available_in_dc)))
            sys.exit(1)
    else:
        candidates = gpu_types  # query failed — try all, let RunPod reject unavailable

    print("  Trying in order: {}".format(", ".join(candidates)))

    for gpu in candidates:
        print("  Trying {}...".format(gpu), end=" ", flush=True)
        try:
            pod = runpod.create_pod(
                name=_POD_NAME,
                image_name=image,
                gpu_type_id=gpu,
                gpu_count=1,
                container_disk_in_gb=container_disk,
                volume_in_gb=volume_disk,
                volume_mount_path=_VOLUME_MOUNT,
                ports=exposed_ports,
                network_volume_id=network_volume_id,
                data_center_id=datacenter_id,
                support_public_ip=True,
                start_ssh=(mode == "ssh"),
            )
            new_pod_id = pod["id"]
            print("ok")
            print("  Created pod: {} ({})".format(new_pod_id, gpu))
            _update_env_key("RUNPOD_POD_ID", new_pod_id)
            return new_pod_id
        except Exception as exc:
            print("unavailable ({})".format(exc))

    print("ERROR: none of the GPUs in RUNPOD_GPU_TYPES have capacity in {} right now.".format(datacenter_id))
    sys.exit(1)


def _wait_for_pipeline(ws_address: str) -> None:
    """Poll the pipeline's TCP port until it accepts connections."""
    host, port_str = ws_address.rsplit(":", 1)
    port = int(port_str)

    print("\nWaiting for pipeline to be ready (up to {}s)...".format(_PIPELINE_TIMEOUT))
    deadline = time.time() + _PIPELINE_TIMEOUT

    while time.time() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            print("  Pipeline is ready.")
            return
        except (socket.error, OSError):
            time.sleep(_POLL_INTERVAL)

    print("ERROR: Pipeline port not reachable after {}s.".format(_PIPELINE_TIMEOUT))
    if _get_deploy_mode() == "ssh":
        print("SSH into the pod and check: tmux attach -t {}".format(_TMUX_SESSION))
    else:
        print("Check pod logs on the RunPod dashboard.")
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
        except paramiko.ssh_exception.SSHException:
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
        except (socket.error, OSError):
            time.sleep(_POLL_INTERVAL)
    print("ERROR: SSH not reachable after {}s.".format(_SSH_TIMEOUT))
    sys.exit(1)


def _ssh_run(client: object, command: str, label: str) -> None:
    """Run a command over SSH, streaming output. Exits on non-zero return code."""
    print("\n[{}] $ {}".format(label, command))
    _, stdout, stderr = client.exec_command(command, get_pty=True)  # type: ignore[attr-defined]
    for line in iter(stdout.readline, ""):
        print("  " + line.rstrip())
    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        err = stderr.read().decode().strip()
        print("ERROR: '{}' failed (exit {}).".format(label, exit_code))
        if err:
            print(err)
        sys.exit(1)


def _ssh_setup_and_start(ssh_address: str, key_path: str) -> None:
    """
    SSH into the pod, clone repo if missing, run startup.sh, start pipeline in tmux.

    All steps are idempotent:
    - git clone only runs if /workspace/Phantom does not exist (first deploy)
    - startup.sh skips venv install if /workspace/venv already exists (subsequent runs)
    """
    paramiko = _require_paramiko()
    host, port_str = ssh_address.rsplit(":", 1)
    port = int(port_str)

    _wait_for_ssh_tcp(host, port)

    key = _load_ssh_key(key_path)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print("  Connecting via SSH...")
    client.connect(hostname=host, port=port, username="root", pkey=key, timeout=30)

    try:
        # Clone repo if not present — only runs on first-ever deploy
        repo_url = os.getenv("RUNPOD_REPO_URL")
        if repo_url:
            _ssh_run(
                client,
                "[ -d {dir} ] || git clone {url} {dir}".format(
                    dir=_REMOTE_PHANTOM_DIR, url=repo_url
                ),
                "git-clone",
            )
        else:
            # Verify the repo exists — if not, give a clear error
            _ssh_run(
                client,
                "[ -d {dir} ] || {{ echo 'ERROR: {dir} not found. Set RUNPOD_REPO_URL in .env to auto-clone.'; exit 1; }}".format(
                    dir=_REMOTE_PHANTOM_DIR
                ),
                "repo-check",
            )

        # Run startup.sh — installs ffmpeg, creates venv on first run
        _ssh_run(client, "bash {}".format(_REMOTE_STARTUP), "startup")

        # Kill any leftover tmux session from a previous run
        _ssh_run(
            client,
            "tmux kill-session -t {} 2>/dev/null || true".format(_TMUX_SESSION),
            "tmux-cleanup",
        )

        # Start pipeline in a detached tmux session using the workspace venv
        pipeline_cmd = "{} {} --execution-provider cuda".format(
            _REMOTE_VENV_PYTHON, _REMOTE_PIPELINE
        )
        _ssh_run(
            client,
            "tmux new-session -d -s {} '{}'".format(_TMUX_SESSION, pipeline_cmd),
            "pipeline-start",
        )
        print("\n  Pipeline started in tmux session '{}'.".format(_TMUX_SESSION))
        print("  To attach: ssh into pod and run: tmux attach -t {}".format(_TMUX_SESSION))
    finally:
        client.close()


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_start(pod_id: Optional[str]) -> None:
    """
    Full boot sequence. Behaviour depends on RUNPOD_DEPLOY_MODE:
      ssh:    resume/deploy → SSH setup → start pipeline in tmux → update .env
      docker: resume/deploy → wait for pipeline (auto-started) → update .env
    """
    mode = _get_deploy_mode()
    print("Deploy mode: {}".format(mode))

    # Step 1 — get a running pod
    active_pod_id: str
    if pod_id:
        pod = runpod.get_pod(pod_id)
        if pod is not None:
            print("Resuming pod {}...".format(pod_id))
            try:
                runpod.resume_pod(pod_id, gpu_count=1)
                active_pod_id = pod_id
            except Exception as exc:
                print("  Resume failed: {}. Deploying a new pod...".format(exc))
                active_pod_id = _deploy_new_pod(mode)
        else:
            print("Pod {} not found. Deploying a new pod...".format(pod_id))
            active_pod_id = _deploy_new_pod(mode)
    else:
        print("No RUNPOD_POD_ID set. Deploying a new pod...")
        active_pod_id = _deploy_new_pod(mode)

    # Step 2 — mode-specific setup
    if mode == "ssh":
        ssh_address, ws_address = _wait_for_ports_ssh(active_pod_id)
        key_path = os.getenv("RUNPOD_SSH_KEY_PATH", "~/.ssh/id_ed25519")
        _ssh_setup_and_start(ssh_address, key_path)
    else:
        ws_address = _wait_for_port_docker(active_pod_id)

    # Step 3 — wait for pipeline to bind port 9000
    _wait_for_pipeline(ws_address)

    # Step 4 — update .env
    _update_env_key("PHANTOM_API_URL", "ws://{}/ws".format(ws_address))

    print("\nDone. Open the desktop:")
    print("  python desktop.py")


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

    ws_addr = _get_port_address(pod_id, 9000)
    if ws_addr:
        print("URL:    ws://{}/ws".format(ws_addr))
    else:
        print("URL:    not available (pod may be stopped or starting)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse args and dispatch to the right command."""
    parser = argparse.ArgumentParser(
        description="Phantom RunPod Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  start      Resume or deploy pod, setup (ssh) or wait (docker), update .env
  stop       Pause pod (preserves /workspace volume — models intact)
  terminate  Permanently delete pod (network volume survives)
  status     Show pod state, GPU, cost, and current WebSocket address

Set RUNPOD_DEPLOY_MODE=ssh (development) or docker (production) in .env.
        """,
    )
    parser.add_argument("command", choices=["start", "stop", "terminate", "status"])
    args = parser.parse_args()

    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set in .env")
        sys.exit(1)
    runpod.api_key = api_key

    pod_id = os.getenv("RUNPOD_POD_ID") or None

    if args.command in ("stop", "terminate", "status"):
        if not pod_id:
            print("ERROR: RUNPOD_POD_ID not set in .env")
            sys.exit(1)
        dispatch = {
            "stop": cmd_stop,
            "terminate": cmd_terminate,
            "status": cmd_status,
        }
        dispatch[args.command](pod_id)  # type: ignore[operator]
    else:
        cmd_start(pod_id)


if __name__ == "__main__":
    main()
