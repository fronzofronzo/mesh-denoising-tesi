import numpy as np
import os
import trimesh
import torch
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.ferLogger(__name__)

try:
    from GCNModel import DGCNN
    import parsers 
    from datautils import loadMAT
    from mesh_reader_optimized import calc_centroid
except ImportError as e:
    logger.error(f"Error importing dependencies: {e}")
    raise

# --- Helper functions --- 

def find_first_ring_neighborhood(vertex_index, mesh):
    """
    Find indices of faces that shares common vertex.

    Args:
        vertex_index (int): vertex index.
        mesh (trimesh.Trimesh): target mesh.
    Returns:
        Indexes array of faces that shares the vertex.
    """
    if vertex_index >= len(mesh.vertices) or vertex_index < 0:
        return np.array([], dtype=int)
    
    try:
        boolean_matrix = mesh_faces == vertex_index
        faces_first_ring = np.any(boolean_matrix, axis=1)
        return np.where(faces_first_ring)[0]
