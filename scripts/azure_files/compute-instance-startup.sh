#!/bin/bash
set -eux  # Exit on error, print commands

echo "Initializing Compute VM at startup..."

# Prevent interactive prompts during package installs
export DEBIAN_FRONTEND=noninteractive

# Logging
exec > >(tee -i /var/log/startup-script.log)
exec 2>&1

# --- Step 1: Install dos2unix ---
echo "Installing dos2unix..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends dos2unix || true

# --- Step 2: Stop DNS / port 53 services ---
echo "Stopping systemd-resolved (dnsmasq on port 53)..."
if systemctl list-unit-files | grep -q systemd-resolved.service; then
    sudo systemctl stop systemd-resolved || true
    echo "systemd-resolved stopped"
else
    echo "systemd-resolved not found, skipping"
fi

echo "Stopping named.service (DNS)..."
if systemctl list-unit-files | grep -q named.service; then
    sudo systemctl stop named || true
    echo "named.service stopped"
else
    echo "named.service not found, skipping"
fi

# --- Step 3: Stop nginx (port 80) ---
echo "Stopping nginx..."
if systemctl list-unit-files | grep -q nginx.service; then
    sudo systemctl stop nginx || true
    echo "nginx.service stopped"
elif command -v service >/dev/null; then
    sudo service nginx stop || true
    echo "nginx.service stopped via service command"
else
    echo "nginx not found, skipping"
fi

echo "===== Startup script finished successfully ====="
