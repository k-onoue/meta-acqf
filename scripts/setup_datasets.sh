#!/bin/bash

# Define directories
DATA_DIR="./data/raw"
mkdir -p $DATA_DIR

echo "=================================================================="
echo "Starting Dataset Setup (Download + Preprocessing)"
echo "=================================================================="
echo "Saving to: $DATA_DIR"
echo ""

# ------------------------------------------------------------------
# 1. Reuters-21578
# ------------------------------------------------------------------
echo "[1/3] Downloading Reuters-21578..."
cd $DATA_DIR
if [ ! -d "reuters21578" ]; then
    wget -c --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz
    mkdir -p reuters21578
    tar -xvzf reuters21578.tar.gz -C reuters21578
    rm reuters21578.tar.gz
    echo "      -> Reuters downloaded."
else
    echo "      -> Reuters directory already exists. Skipping."
fi
cd ../..
echo ""

# ------------------------------------------------------------------
# 2. NeurIPS (NIPS 1-12) - FIXED
# ------------------------------------------------------------------
echo "[2/3] Downloading NeurIPS 1-12..."
echo "      Source: Sam Roweis (via Internet Archive)"
cd $DATA_DIR

# Use Wayback Machine snapshots to bypass the 403 Forbidden error
# Snapshot date: 2016 (stable version)
NIPS_MAT_URL="https://web.archive.org/web/20160303191350/http://cs.nyu.edu/~roweis/data/nips12.mat"
NIPS_RAW_URL="https://web.archive.org/web/20160303230438/http://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz"

wget -c -O nips12.mat "$NIPS_MAT_URL"
wget -c -O nips12raw_str602.tgz "$NIPS_RAW_URL"

if [ -f "nips12raw_str602.tgz" ]; then
    tar -xvzf nips12raw_str602.tgz
    rm nips12raw_str602.tgz
    echo "      -> NeurIPS downloaded successfully."
else
    echo "      -> Error downloading NeurIPS."
fi
cd ../..
echo ""

# ------------------------------------------------------------------
# 3. WebKB (4 Universities)
# ------------------------------------------------------------------
echo "[3/3] Downloading WebKB (4 Universities)..."
cd $DATA_DIR
if [ ! -d "webkb" ]; then
    # Note: Using Carnegie Mellon's server. If this fails, we can add a mirror.
    wget -c http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz
    
    # Extract
    # Note: This tarball extracts into a folder named "webkb" automatically
    tar -xvzf webkb-data.gtar.gz
    rm webkb-data.gtar.gz
    echo "      -> WebKB downloaded."
else
    echo "      -> WebKB directory already exists. Skipping."
fi
cd ../..

echo "=================================================================="
echo "Download Complete."
echo "=================================================================="
echo ""

# ------------------------------------------------------------------
# 4. Preprocessing
# ------------------------------------------------------------------
echo "[4/4] Running preprocessing..."
python scripts/preprocess_datasets.py

echo ""
echo "=================================================================="
echo "Setup Complete. Processed datasets saved to data/"
echo "=================================================================="

