#!/bin/bash
set -eux  # Exit immediately if a command fails, print each command

# Prevent interactive prompts during apt installs
export DEBIAN_FRONTEND=noninteractive

# Logging for debugging
exec > >(tee -i /var/log/startup-script.log)
exec 2>&1

echo "===== Azure ML Compute Instance Startup Script ====="

# --- Step 1: Fix duplicate apt sources that cause warnings ---
echo "Cleaning up duplicate apt sources..."
sudo rm -f /etc/apt/sources.list.d/archive_uri-https_packages_microsoft_com_ubuntu_22_04_prod-jammy.list || true
sudo rm -f /etc/apt/sources.list.d/azure-cli.sources || true

# --- Step 2: Update and upgrade system packages ---
echo "Updating apt package lists..."
sudo apt-get update -y
sudo apt-get upgrade -y || true  # don't fail if upgrade is partially blocked

# --- Step 3: Install required packages ---
echo "Installing required packages..."
sudo apt-get install -y --no-install-recommends dos2unix

# --- Step 4: Stop conflicting services if they exist ---
echo "Checking for named.service..."
if systemctl list-unit-files | grep -q named.service; then
    sudo systemctl stop named || true
    echo "named.service stopped."
else
    echo "named.service not found, skipping."
fi

echo "Checking for nginx.service..."
if systemctl list-unit-files | grep -q nginx.service; then
    sudo systemctl stop nginx || true
    echo "nginx.service stopped."
else
    echo "nginx.service not found, skipping."
fi

# --- Step 5: Final cleanup ---
echo "Cleaning up..."
sudo apt-get autoremove -y || true
sudo apt-get clean

echo "===== Startup script finished successfully ====="
