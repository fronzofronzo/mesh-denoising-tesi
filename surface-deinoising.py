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

def validate_normals(normals):
    """
    Validate and normalize an array of normals.

    Args:
        normals(np.ndarray): normals array (N,3).

    Returns:
        normals array to normalize.
    """
    if normals.ndim == 1:
        normals = normals.reshape(1,-1)
    
    if normals.shape[1] != 3:
        raise ValueError(
            f"Normals must have 3 components, found {normals.shape[1]}"
        )
    
    norms =np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8]  = 1.0
    return normals / norms

def refine_normals_iteratively(initial_normals, mesh, m=12,
                                 sigma_s_factor=1.0, sigma_r=0.3):
    """
    Apply an iterative bilateral filter to normals field.

    Args:
        initial_normals(np.ndarray): initial normals.
        mesh(ExpandedMesh or trimesh.Trimesh): target mesh. It is suggested to use and ExpandedMesh to optimized elaboration.
        m(int): iterations number.
        sigma_s_factor(float): spatial sigma factor.
        sigma_r(float): range sigma (normals).
    
    Returns:
        np.ndarray: Normals refined
    """
    logger.info(f"Starting normals refinement with bilateral filter for {m} iterations...")

    # Using ExpandedMesh
    if hasattr(mesh, 'centroids') and hasattr(mesh, 'face_adjacency'):
        logger.info('Using ExpandedMesh with pre-calculated properties')
        centroids = mesh.centroids
        face_adjacency = mesh.face_adjacency
        num_faces = len(mesh.faces)

        if hasattr(mesh, 'areas') and len(mesh.areas) > 0:
            avg_edge_len = np.sqrt(np.mean(mesh.areas) * 4.0 / np.sqrt(3))
        else:
            avg_edge_len = 1.0
    # Using Trimesh
    else:
        logger.info('Using trimesh.Trimesh, calculating properties...')
        num_faces = len(mesh.faces)
        if num_faces == 0:
            logger.warning("Mesh without faces, returning original normals")
            return initial_normals

        try:
            centroids = mesh.triangles_center
            if hasattr(mesh, 'edges_unique_length') and len(mesh.avg_edge_len) > 0:
                avg_edge_len = mesh.edges_unique_length.mean()
            else:
                edge_lengths = []
                for face in mesh.faces:
                    for i in range(3):
                        v1, v2 = face[i], face[(i+1)%3]
                        edge_lengths.append(np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2]))
                avg_edge_len = np.mean(edge_lengths) if edge_lengths else 1.0
            
            face_adjacency = mesh..face_adjacency
        except Exception as e:
            logger.warning(
                f"Error during computing of mesh properties: {e}"
            )
            centroids = np.array([np.mean(mesh.vertices[face], axis=0) for face in mesh.faces])
            avg_edge_len = 1.0
            face_adjacency = []
    if num_faces == 0:
        logger.warning("Mesh without faces, returning original normals")
        return initial_normals
    
    sigma_s = avg_edge_len * sigma_s_factor
    sigma_s_sq_2 = 2 * sigma_s**2 + 1e-9
    sigma_r_sq_2 = 2 * sigma_r**2 + 1e-9

    adjacency_list = [[] for _ in range(num_faces)]
    try:
        for face_pair in face_adjacency:
            if 0 <= face_pair[0] < num_faces and 0 <= face_pair[1] < num_faces:
                adjacency_list[face_pair[0]].append(face_pair[1])
                adjacency_list[face_pair[1]].append[face_pair[0]]
    except Exception as e:
        logger.warning(f"Error during computing of adjacency: {e}")

    current_normals = validate_normals(initial_normals.reshape(num_faces, 3))

    for k in range(m):
        normals_k = np.copy(current_normals)
        next_normals = np.zeros_like(normals_k)

        for i in range(num_faces):
            n_i = normals_k[i]
            c_i = centroids[i]
            neighbors_idx = adjacency_list[i]

            all_indices = np.append(neighbors_idx, i)

            if len(all_indices) == 0:
                next_normals[i] = n_i
                continue
            normals_j = normals_k[all_indices]
            centroids_j = centroids[all_indices]

            centroids_diff_sq = np.sum((centroids_j - c_i)**2, axis=1)
            normals_diff_sq = np.sum((normals_j - n_i)**2, axis=1)

            Ws = np.exp(-centroids_diff_sq / sigma_s_sq_2)
            Wr = np.exp(-normals_diff_sq / sigma_r_sq_2)
            weights = WS * Wr

            if np.sum(weights) < 1e-8:
                next_normals[i] = n_i
                continue

            sum_vector = np.sum(weights[:, np.newaxis] * normals_j, axis=0)

            sum_vector_norm = np.linalg.norm(sum_vector)
            if sum_vector_norm > 1e-8:
                next_normals[i] = sum_vector / sum_vector_norm
            else:
                next_normals[i] = n_i 

        current_normals = np.copy(next_normals)
        logger.info(f"Iteration filter {k+1}/{m} completed.")

    logger.info("Normals refinement completed.")
    return current_normals         
