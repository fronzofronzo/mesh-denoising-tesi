import trimesh
import numpy as np
import os

def calculate_average_edge_length(mesh):
    """Compute average length of uniques edges of mesh."""
    edges = mesh.edges_unique
    vertices = mesh.vertices
    
    edge_lengths = np.linalg.norm(vertices[edges[:,0]] - vertices[edges[:,1]], axis=1)
    average_length = np.mean(edge_lengths) if len(edge_lengths) > 0 else 0
    return average_length

def add_gaussian_noise_paper(mesh, noise_level):
    vertices = mesh.vertices.copy()
    num_vertices = vertices.shape[0]
    
    # Compute sigma 
    avg_edge_length = calculate_average_edge_length(mesh)
    if avg_edge_length == 0:
        avg_edge_length = mesh.scale
        if avg_edge_length == 0: avg_edge_length = 1.0
        
    sigma = avg_edge_length * noise_level
    
    # Generate and add gaussian noise 
    noise = np.random.randn(num_vertices,3) * sigma
    noisy_vertices = vertices + noise
    
    noisy_mesh = trimesh.Trimesh(vertices=noisy_vertices, faces=mesh.faces)    
    return noisy_mesh
