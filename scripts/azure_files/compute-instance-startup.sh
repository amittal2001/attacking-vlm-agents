#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# Redirect logs
exec > >(tee -i /var/log/startup-script.log)
exec 2>&1

echo "[INFO] Cleaning up old sources..."
sudo rm -f /etc/apt/sources.list.d/archive_uri-https_packages_microsoft_com_ubuntu_22_04_prod-jammy.list
sudo rm -f /etc/apt/sources.list.d/azure-cli.sources

echo "[INFO] Fixing TensorFlow Serving repo key..."
# Remove old key if exists
sudo rm -f /usr/share/keyrings/tf-serving.gpg
# Import new key
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo gpg --dearmor -o /usr/share/keyrings/tf-serving.gpg

# Ensure correct repo list
cat <<EOF | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
deb [signed-by=/usr/share/keyrings/tf-serving.gpg] https://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal
EOF

echo "[INFO] Running apt-get update..."
sudo apt-get update -y || true   # allow non-critical repo failures

echo "[INFO] Installing required packages..."
sudo apt-get install -y --no-install-recommends dos2unix

echo "[INFO] Trying to stop named.service if present..."
if systemctl list-unit-files | grep -q named.service; then
    sudo systemctl stop named.service || true
else
    echo "[INFO] named.service not found, skipping."
fi

echo "[INFO] Startup script completed successfully."
