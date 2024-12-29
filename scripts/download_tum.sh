"""
Downloads the TUM-RGBD dataset to $DATA_ROOT/datasets/TUM_RGBD/

Usage:
    sh scripts/download_tum.sh $DATA_ROOT
"""

DATA_ROOT=$1
if [ -z "$DATA_ROOT" ]; then
    DATA_ROOT="."
fi
mkdir -p $DATA_ROOT
cd $DATA_ROOT

mkdir -p datasets/TUM_RGBD
cd datasets/TUM_RGBD
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xvzf rgbd_dataset_freiburg1_desk.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz
tar -xvzf rgbd_dataset_freiburg1_desk2.tgz
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
tar -xvzf rgbd_dataset_freiburg1_room.tgz
wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz
tar -xvzf rgbd_dataset_freiburg2_xyz.tgz
wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
tar -xvzf rgbd_dataset_freiburg3_long_office_household.tgz
