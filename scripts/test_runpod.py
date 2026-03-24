#!/usr/bin/env python3
"""
Quick RunPod connectivity and setup test.

Verifies:
1. RUNPOD_API_KEY is set and valid
2. Can list available GPU types
3. 8xH100 SXM is available
4. Can create a minimal pod, SSH into it, and terminate it

Usage:
    python scripts/test_runpod.py              # API check only (free)
    python scripts/test_runpod.py --full       # Full test: create pod, SSH, terminate (~$1)
"""

import argparse
import os
import subprocess
import sys
import time


def test_api():
    """Test API key and list GPU availability."""
    import runpod

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("FAIL: RUNPOD_API_KEY not set")
        return False
    runpod.api_key = api_key

    print("1. Testing API key...")
    try:
        gpus = runpod.get_gpus()
        print(f"   OK: API key valid, {len(gpus)} GPU types available")
    except Exception as e:
        print(f"   FAIL: {e}")
        return False

    print("2. Checking H100 SXM availability...")
    h100_found = False
    for gpu in gpus:
        if "H100" in gpu.get("id", "") and "SXM" in gpu.get("id", "").upper():
            print(f"   Found: {gpu['id']}")
            h100_found = True
        elif "H100" in gpu.get("displayName", ""):
            print(f"   Found: {gpu.get('displayName', gpu['id'])}")
            h100_found = True

    if not h100_found:
        # Try broader search
        for gpu in gpus:
            name = gpu.get("displayName", gpu.get("id", ""))
            if "H100" in name:
                print(f"   Found: {name} (id: {gpu['id']})")
                h100_found = True

    if not h100_found:
        print("   WARNING: H100 SXM not found in available GPUs")
        print("   Available GPU types with 'H100':")
        for gpu in gpus:
            name = gpu.get("displayName", gpu.get("id", ""))
            if "H100" in name.upper():
                print(f"     - {name} (id: {gpu['id']})")
        if not any("H100" in gpu.get("displayName", gpu.get("id", "")).upper() for gpu in gpus):
            print("   No H100 GPUs found at all. Listing all types:")
            for gpu in gpus[:10]:
                print(f"     - {gpu.get('displayName', gpu['id'])}")
            print(f"     ... and {len(gpus)-10} more")
    else:
        print("   OK")

    print("3. Checking SSH key...")
    ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
    rsa_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
    if os.path.exists(ssh_key_path):
        print(f"   OK: Found {ssh_key_path}")
    elif os.path.exists(rsa_key_path):
        print(f"   OK: Found {rsa_key_path}")
    else:
        print("   WARNING: No SSH public key found at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub")
        print("   You may need to set up SSH keys for pod access")

    return True


def test_full():
    """Full test: create a cheap pod, SSH in, terminate."""
    import runpod

    api_key = os.environ.get("RUNPOD_API_KEY")
    runpod.api_key = api_key

    print("\n4. Full test: creating a minimal pod...")
    print("   (This will cost ~$0.50-1.00)")

    # Prefer cheap/common GPUs for connectivity testing; order by typical cost
    gpu_candidates = [
        "NVIDIA GeForce RTX 3070",
        "NVIDIA GeForce RTX 3080",
        "NVIDIA GeForce RTX 3080 Ti",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 3090 Ti",
        "NVIDIA RTX A4000",
        "NVIDIA RTX A4500",
        "NVIDIA RTX A5000",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A6000",
        "NVIDIA L4",
        "NVIDIA L40",
        "NVIDIA A40",
    ]

    pod_id = None
    try:
        # Read SSH public key to inject into pod
        ssh_pub_key = None
        for key_path in ["~/.ssh/id_ed25519.pub", "~/.ssh/id_rsa.pub"]:
            path = os.path.expanduser(key_path)
            if os.path.exists(path):
                with open(path) as f:
                    ssh_pub_key = f.read().strip()
                break
        if not ssh_pub_key:
            print("   FAIL: No SSH public key found — cannot authenticate to pod")
            return False

        pod = None
        for gpu_type in gpu_candidates:
            for cloud_type in ("COMMUNITY", "SECURE"):
                try:
                    print(f"   Trying {gpu_type} ({cloud_type})...")
                    pod = runpod.create_pod(
                        name="pgolf-connectivity-test",
                        image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                        gpu_type_id=gpu_type,
                        gpu_count=1,
                        cloud_type=cloud_type,
                        container_disk_in_gb=10,
                        support_public_ip=True,
                        ports="22/tcp",
                        env={"PUBLIC_KEY": ssh_pub_key},
                    )
                    break
                except Exception:
                    continue
            if pod:
                break

        if not pod:
            print("   FAIL: No available GPU found across all candidates")
            return False

        pod_id = pod["id"]
        print(f"   Pod created: {pod_id}")

        # Wait for ready
        print("   Waiting for pod to start...")
        start = time.time()
        ssh_ip = None
        ssh_port = None
        while time.time() - start < 180:
            pod_info = runpod.get_pod(pod_id)
            if not pod_info:
                time.sleep(5)
                continue
            status = pod_info.get("desiredStatus", "?")
            uptime = pod_info.get("uptimeSeconds", 0)
            runtime = pod_info.get("runtime", {})
            ports = (runtime or {}).get("ports", [])
            print(f"   status={status} uptime={uptime}s ports={ports}")
            for p in (ports or []):
                if p.get("privatePort") == 22:
                    ssh_ip = p.get("ip")
                    ssh_port = p.get("publicPort")
                    break
            if ssh_ip and ssh_port:
                break
            time.sleep(5)

        if not ssh_ip:
            print("   FAIL: Pod did not become ready in 180s")
            return False

        print(f"   Pod ready: {ssh_ip}:{ssh_port}")

        # Test SSH
        print("   Testing SSH connection...")
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=15",
                "-p", str(ssh_port),
                f"root@{ssh_ip}",
                "echo 'SSH OK' && nvidia-smi --query-gpu=name --format=csv,noheader | head -1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print(f"   OK: {result.stdout.strip()}")
        else:
            print(f"   FAIL: SSH failed: {result.stderr[:200]}")
            return False

        print("   Full connectivity test PASSED")
        return True

    except Exception as e:
        print(f"   FAIL: {e}")
        return False

    finally:
        if pod_id:
            print(f"   Terminating test pod {pod_id}...")
            try:
                runpod.terminate_pod(pod_id)
                print("   Pod terminated.")
            except Exception as e:
                print(f"   WARNING: Failed to terminate: {e}")
                print(f"   MANUALLY TERMINATE pod {pod_id}!")


def main():
    parser = argparse.ArgumentParser(description="Test RunPod connectivity")
    parser.add_argument("--full", action="store_true",
                        help="Full test: create pod, SSH, terminate (~$1)")
    args = parser.parse_args()

    ok = test_api()
    if not ok:
        sys.exit(1)

    if args.full:
        ok = test_full()
        if not ok:
            sys.exit(1)
    else:
        print("\nBasic checks passed. Run with --full to test pod creation + SSH (~$1).")


if __name__ == "__main__":
    main()
