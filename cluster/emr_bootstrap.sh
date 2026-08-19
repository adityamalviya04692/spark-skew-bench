#!/usr/bin/env bash
# EMR bootstrap action for skewbench.
# Add via: --bootstrap-actions Path=s3://<bucket>/emr_bootstrap.sh
set -euo pipefail

sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install "numpy>=1.26" "pandas>=2.0" "pyarrow>=14.0" "PyYAML>=6.0"

# Event logging must be on for the metrics layer to work at all: skewbench
# derives every measurement from the event log rather than from a listener, so
# that a run can be re-analysed later without being repeated.
sudo mkdir -p /var/log/spark/skewbench
sudo chmod 777 /var/log/spark/skewbench
