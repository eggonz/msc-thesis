"""
Downloads the Replica meshes to $DATA_ROOT/datasets/cull_replica_mesh/

Usage:
    sh scripts/download_cull_replica_mesh.sh $DATA_ROOT
"""

DATA_ROOT=$1
if [ -z "$DATA_ROOT" ]; then
    DATA_ROOT="."
fi
mkdir -p $DATA_ROOT
cd $DATA_ROOT

wget https://cvg-data.inf.ethz.ch/nice-slam/cull_replica_mesh.zip
unzip cull_replica_mesh.zip