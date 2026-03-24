#!/usr/bin/env python3
"""
RunPod 8xH100 evaluation runner for Parameter Golf.

Creates a pod, uploads the training script, runs training, downloads results,
and terminates the pod. Returns structured JSON results.

Usage:
    python scripts/runpod_eval.py [--dry-run] [--description DESC] [--seed SEED]

Requires:
    - RUNPOD_API_KEY environment variable
    - pip install runpod paramiko
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKING_SCRIPT = REPO_ROOT / "working" / "train_gpt.py"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
BUDGET_FILE = REPO_ROOT / "BUDGET.md"

# RunPod configuration
TEMPLATE_ID = "y5cejece4j"  # Official Parameter Golf template
GPU_TYPE_ID = "NVIDIA H100 80GB HBM3"
GPU_COUNT = 8
CLOUD_TYPE = "SECURE"
CONTAINER_DISK_GB = 50
POD_NAME_PREFIX = "pgolf-eval"

# Cost estimate: 8xH100 SXM at ~$2.69/GPU/hr = ~$21.52/hr
COST_PER_HOUR = 21.52

# Timeouts
POD_READY_TIMEOUT = 900  # 15 min to become ready (8xH100 can be slow)
TRAINING_TIMEOUT = 900   # 15 min max for training (10 min + buffer)
SETUP_TIMEOUT = 300      # 5 min for setup commands


def get_budget_spent() -> float:
    """Parse BUDGET.md to get cumulative spend."""
    if not BUDGET_FILE.exists():
        return 0.0
    text = BUDGET_FILE.read_text()
    # Parse last data row of the markdown table: last column is cumulative
    matches = re.findall(r"\|\s*\$?([\d.]+)\s*\|?\s*$", text, re.MULTILINE)
    return float(matches[-1]) if matches else 0.0


def append_budget_entry(description: str, duration_min: float, cost: float):
    """Append a line to BUDGET.md."""
    cumulative = get_budget_spent() + cost
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"| {timestamp} | {description} | {duration_min:.1f} min | ${cost:.2f} | ${cumulative:.2f} |\n"

    if not BUDGET_FILE.exists():
        BUDGET_FILE.write_text(
            "# RunPod Budget Tracker\n\n"
            "| Timestamp | Description | Duration | Cost | Cumulative |\n"
            "|-----------|-------------|----------|------|------------|\n"
            + entry
        )
    else:
        with open(BUDGET_FILE, "a") as f:
            f.write(entry)


def wait_for_pod_ready(pod_id: str, timeout: int = POD_READY_TIMEOUT) -> dict:
    """Poll until pod is in RUNNING state."""
    import runpod

    start = time.time()
    while time.time() - start < timeout:
        pod = runpod.get_pod(pod_id)
        if not pod:
            time.sleep(10)
            continue
        status = pod.get("desiredStatus", "unknown")
        runtime = pod.get("runtime", {})

        ports = (runtime or {}).get("ports", [])
        ssh_ip = None
        ssh_port = None
        for p in (ports or []):
            if p.get("privatePort") == 22:
                ssh_ip = p.get("ip")
                ssh_port = p.get("publicPort")
                break
        if ssh_ip and ssh_port:
            print(f"Pod ready: {ssh_ip}:{ssh_port}")
            return {"pod": pod, "ssh_ip": ssh_ip, "ssh_port": ssh_port}

        print(f"  Waiting for pod... status={status}, elapsed={time.time()-start:.0f}s")
        time.sleep(10)

    raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")


def run_ssh_command(ssh_ip: str, ssh_port: int, command: str,
                    timeout: int = TRAINING_TIMEOUT) -> tuple[int, str, str]:
    """Run a command on the pod via SSH and return (exit_code, stdout, stderr)."""
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        "-p", str(ssh_port),
        f"root@{ssh_ip}",
        command,
    ]
    print(f"  SSH: {command[:120]}...")
    result = subprocess.run(
        ssh_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def scp_upload(ssh_ip: str, ssh_port: int, local_path: str, remote_path: str):
    """Upload a file via SCP."""
    cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-P", str(ssh_port),
        local_path,
        f"root@{ssh_ip}:{remote_path}",
    ]
    subprocess.run(cmd, check=True, timeout=60)


def scp_download(ssh_ip: str, ssh_port: int, remote_path: str, local_path: str):
    """Download a file via SCP."""
    cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-P", str(ssh_port),
        f"root@{ssh_ip}:{remote_path}",
        local_path,
    ]
    subprocess.run(cmd, check=True, timeout=120)


def parse_training_log(log_text: str) -> dict:
    """Extract key metrics from training output."""
    results = {
        "val_loss": None,
        "val_bpb": None,
        "artifact_size_bytes": None,
        "train_time_ms": None,
        "steps": None,
    }

    # Look for final int8 roundtrip results (the actual submission metrics)
    for line in log_text.splitlines():
        if "final_int8_zlib_roundtrip" in line and "val_bpb:" in line:
            m = re.search(r"val_loss:([\d.]+)", line)
            if m:
                results["val_loss"] = float(m.group(1))
            m = re.search(r"val_bpb:([\d.]+)", line)
            if m:
                results["val_bpb"] = float(m.group(1))

        if "Total submission size int8+zlib:" in line:
            m = re.search(r"(\d+) bytes", line)
            if m:
                results["artifact_size_bytes"] = int(m.group(1))

        if "stopping_early:" in line or ("step:" in line and "train_time:" in line):
            m = re.search(r"train_time:(\d+)ms", line)
            if m:
                results["train_time_ms"] = int(m.group(1))
            m = re.search(r"step:(\d+)/", line)
            if m:
                results["steps"] = int(m.group(1))

    return results


def run_evaluation(description: str = "eval", seed: int = 1337, dry_run: bool = False) -> dict:
    """Run a full evaluation cycle on RunPod 8xH100."""
    import runpod

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    runpod.api_key = api_key

    if not WORKING_SCRIPT.exists():
        print(f"ERROR: Working script not found: {WORKING_SCRIPT}")
        sys.exit(1)

    # Check budget
    spent = get_budget_spent()
    print(f"Budget spent so far: ${spent:.2f}")
    if spent > 900:
        print("WARNING: Approaching budget limit ($900+). Proceed with caution.")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_{description}"
    exp_dir = EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save the script being evaluated
    import shutil
    shutil.copy2(WORKING_SCRIPT, exp_dir / "train_gpt.py")

    if dry_run:
        print("DRY RUN: Would create pod, run training, and terminate.")
        print(f"  GPU: {GPU_COUNT}x {GPU_TYPE_ID}")
        print(f"  Template: {TEMPLATE_ID}")
        print(f"  Script: {WORKING_SCRIPT}")
        print(f"  Experiment dir: {exp_dir}")
        return {"dry_run": True, "exp_dir": str(exp_dir)}

    pod_id = None
    start_time = time.time()

    try:
        # Create pod
        print(f"Creating {GPU_COUNT}x H100 pod...")
        ssh_pub_key = None
        for key_path in ["~/.ssh/id_ed25519.pub", "~/.ssh/id_rsa.pub"]:
            path = os.path.expanduser(key_path)
            if os.path.exists(path):
                with open(path) as f:
                    ssh_pub_key = f.read().strip()
                break
        if not ssh_pub_key:
            print("ERROR: No SSH public key found at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub")
            sys.exit(1)

        pod = runpod.create_pod(
            name=f"{POD_NAME_PREFIX}-{timestamp}",
            gpu_type_id=GPU_TYPE_ID,
            gpu_count=GPU_COUNT,
            cloud_type=CLOUD_TYPE,
            container_disk_in_gb=CONTAINER_DISK_GB,
            template_id=TEMPLATE_ID,
            support_public_ip=True,
            env={"PUBLIC_KEY": ssh_pub_key},
        )
        pod_id = pod["id"]
        print(f"Pod created: {pod_id}")

        # Wait for ready
        info = wait_for_pod_ready(pod_id)
        ssh_ip = info["ssh_ip"]
        ssh_port = info["ssh_port"]

        # Setup: clone repo and download data
        print("Setting up environment...")
        setup_cmds = [
            # Clone or pull - handle both fresh and existing states
            "cd /workspace && if [ -d parameter-golf/.git ]; then cd parameter-golf && git pull; else rm -rf parameter-golf; git clone https://github.com/openai/parameter-golf.git; fi",
            "cd /workspace/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024",
            # zstandard is auto-installed by train_gpt.py if needed
        ]
        for cmd in setup_cmds:
            rc, stdout, stderr = run_ssh_command(ssh_ip, ssh_port, cmd, timeout=SETUP_TIMEOUT)
            # Print setup command output for debugging
            if stdout.strip():
                for line in stdout.strip().split('\n')[-5:]:
                    print(f"    {line}")
            if rc != 0:
                print(f"  Setup command failed (rc={rc}): {stderr[:500]}")
                if "No such file" in stderr or "not a git repository" in stderr:
                    raise RuntimeError(f"Critical setup failure: {stderr[:200]}")

        # Upload our modified training script
        print("Uploading training script...")
        scp_upload(ssh_ip, ssh_port, str(WORKING_SCRIPT),
                   "/workspace/parameter-golf/train_gpt.py")

        # Run training
        print(f"Starting training (seed={seed})...")
        train_cmd = (
            f"cd /workspace/parameter-golf && "
            f"SEED={seed} "
            f"RUN_ID=eval_{timestamp} "
            f"DATA_PATH=./data/datasets/fineweb10B_sp1024/ "
            f"TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model "
            f"VOCAB_SIZE=1024 "
            f"torchrun --standalone --nproc_per_node={GPU_COUNT} train_gpt.py 2>&1"
        )
        rc, stdout, stderr = run_ssh_command(ssh_ip, ssh_port, train_cmd, timeout=TRAINING_TIMEOUT)

        # Save full log
        log_text = stdout + "\n" + stderr
        (exp_dir / "train.log").write_text(log_text)
        print(f"Training finished (rc={rc})")

        # Parse results
        results = parse_training_log(log_text)
        results["exit_code"] = rc
        results["seed"] = seed
        results["description"] = description
        results["experiment"] = exp_name
        results["pod_id"] = pod_id

        # Try to download model artifact
        try:
            scp_download(ssh_ip, ssh_port,
                         "/workspace/parameter-golf/final_model.int6.ptz",
                         str(exp_dir / "final_model.int6.ptz"))
        except Exception as e:
            print(f"  Could not download model artifact: {e}")

        # Calculate cost
        elapsed_min = (time.time() - start_time) / 60
        cost = (elapsed_min / 60) * COST_PER_HOUR
        results["elapsed_min"] = round(elapsed_min, 1)
        results["estimated_cost"] = round(cost, 2)

        # Save results
        (exp_dir / "results.json").write_text(json.dumps(results, indent=2))

        # Update budget
        append_budget_entry(description, elapsed_min, cost)

        # Print summary
        print("\n" + "=" * 60)
        print(f"RESULTS: {exp_name}")
        print(f"  val_bpb:  {results.get('val_bpb', 'N/A')}")
        print(f"  val_loss: {results.get('val_loss', 'N/A')}")
        print(f"  artifact: {results.get('artifact_size_bytes', 'N/A')} bytes")
        print(f"  steps:    {results.get('steps', 'N/A')}")
        print(f"  time:     {elapsed_min:.1f} min")
        print(f"  cost:     ${cost:.2f}")
        print(f"  total $:  ${get_budget_spent():.2f}")
        print("=" * 60)

        return results

    except Exception as e:
        print(f"ERROR: {e}")
        elapsed_min = (time.time() - start_time) / 60
        cost = (elapsed_min / 60) * COST_PER_HOUR
        error_result = {
            "error": str(e),
            "elapsed_min": round(elapsed_min, 1),
            "estimated_cost": round(cost, 2),
        }
        (exp_dir / "results.json").write_text(json.dumps(error_result, indent=2))
        append_budget_entry(f"{description} (FAILED)", elapsed_min, cost)
        return error_result

    finally:
        # Always terminate the pod
        if pod_id:
            print(f"Terminating pod {pod_id}...")
            try:
                runpod.terminate_pod(pod_id)
                print("Pod terminated.")
            except Exception as e:
                print(f"WARNING: Failed to terminate pod {pod_id}: {e}")
                print("MANUAL CLEANUP MAY BE REQUIRED!")


def main():
    parser = argparse.ArgumentParser(description="Run Parameter Golf evaluation on RunPod 8xH100")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without creating a pod")
    parser.add_argument("--description", "-d", default="eval", help="Short description for this run")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    results = run_evaluation(
        description=args.description,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    # Output JSON to stdout for machine parsing
    print("\n---JSON_RESULTS---")
    print(json.dumps(results))

    # Exit with appropriate code
    if results.get("error"):
        sys.exit(1)
    if results.get("val_bpb") is None and not results.get("dry_run"):
        sys.exit(1)


if __name__ == "__main__":
    main()
