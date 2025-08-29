#!/bin/bash
set -eux
export DEBIAN_FRONTEND=noninteractive

# Logging
exec > >(tee -i /var/log/startup-script.log)
exec 2>&1

# Fix duplicate apt sources
sudo rm -f /etc/apt/sources.list.d/archive_uri-https_packages_microsoft_com_ubuntu_22_04_prod-jammy.list
sudo rm -f /etc/apt/sources.list.d/azure-cli.sources

sudo apt-get update -y
sudo apt-get install -y --no-install-recommends dos2unix

# Only stop named.service if it exists
if systemctl list-unit-files | grep -q named.service; then
    sudo systemctl stop named.service
fi

echo "Startup script finished successfully.