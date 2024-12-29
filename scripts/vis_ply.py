"""
Visualize a PLY file using Open3D.

Usage:
    python vis_ply.py <path_to_ply_file>
"""

import sys

import open3d as o3d

filepath = sys.argv[1]
mesh = o3d.io.read_triangle_mesh(filepath)
o3d.visualization.draw_geometries([mesh])
