"""
Downloads the Replica dataset to $DATA_ROOT/datasets/Replica/

Usage:
    sh scripts/download_replica.sh $DATA_ROOT
"""

DATA_ROOT=$1
if [ -z "$DATA_ROOT" ]; then
    DATA_ROOT="."
fi
mkdir -p $DATA_ROOT
cd $DATA_ROOT

mkdir -p datasets
cd datasets
# you can also download the Replica.zip manually through
# link: https://caiyun.139.com/m/i?1A5Ch5C3abNiL password: v3fY (the zip is split into smaller zips because of the size limitation of caiyun)
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
