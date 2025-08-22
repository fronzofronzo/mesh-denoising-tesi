#!/usr/bin/env python3
"""
Versione ottimizzata del main script con focus sulla ricerca dei vicini
"""

import trimesh
import numpy as np
import scipy.io
import os
import argparse
from AdjacencyGraph import AdjacencyGraph
from joblib import Parallel, delayed
from generate_noise import add_gaussian_noise_paper
from ExpandedMesh import ExpandedMesh 
from patch_generator_mod_optimized import (
    PatchData, 
    process_patch_optimized
)

def calc_normal(face, vertices):
    """Calcola la normale di una faccia"""
    p1 = vertices[face[0]]
    p2 = vertices[face[1]] 
    p3 = vertices[face[2]]
    
    v = p2 - p1
    w = p3 - p1
    
    return np.cross(v, w)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Generate patches for selected mesh (OPTIMIZED)")
    parser.add_argument('mesh_name', default=None,
                        help="Mesh to which generate noise",
                        type=str)
    parser.add_argument('noise_level', default="0.1",
                        help="Select noise level for mesh",
                        type=float)
    parser.add_argument('--filter_iterations', default=1, type=int,
                        help="Number of normal filtering iterations for alpha^n calculation")
    parser.add_argument('--benchmark', action='store_true',
                        help="Run benchmark to compare original vs optimized methods")
    parser.add_argument('--n_jobs', default=-1, type=int,
                        help="Number of parallel jobs (-1 for all cores)")
    
    args = parser.parse_args()
    mesh_name = args.mesh_name
    noise_level = args.noise_level
    filter_iterations = args.filter_iterations
    
    # Carica i mesh
    model_path = os.path.join(script_dir, "testing_models", f"{mesh_name}_gt.obj")
    noised_model_path = os.path.join(script_dir, "testing_models", f"{mesh_name}_noised_{noise_level}_Gaussian.obj")
    
    print(f"Caricamento mesh: {mesh_name}")
    noised_mesh = trimesh.load_mesh(noised_model_path)
    mesh = trimesh.load_mesh(model_path)
    
    print(f"Mesh caricato - Facce: {len(mesh.faces)}, Vertici: {len(mesh.vertices)}")
    
    # Calcola ground truth normals
    print("Calcolo ground truth normals...")
    gt_normals = []
    for i, face in enumerate(mesh.faces):
        gt_normals.append(calc_normal(face, mesh.vertices))
    gt_normals = np.array(gt_normals)
    
    # Crea expanded mesh
    print("Creazione expanded mesh...")
    extended_noised_mesh = ExpandedMesh(noised_mesh)
    
    # Crea grafo di adiacenza condiviso (OTTIMIZZAZIONE CHIAVE)
    print("Costruzione grafo di adiacenza ottimizzato...")
    shared_adjacency_graph = AdjacencyGraph(extended_noised_mesh.face_adjacency)
    print(f"Grafo costruito - Facce: {shared_adjacency_graph.num_faces}")
    
    # Setup directory di output
    num_faces = len(noised_mesh.faces)
    output_directory = os.path.join(script_dir, "new_testing_samples", f"{mesh_name}_{noise_level}")
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Directory di output: {output_directory}")
    print(f"Processando {num_faces} facce con {args.n_jobs} jobs...")
    
    # Processing parallelo ottimizzato
    # NOTA: Il grafo di adiacenza condiviso riduce drasticamente il tempo di inizializzazione
    results = Parallel(n_jobs=args.n_jobs, backend='loky', verbose=10)(
        delayed(process_patch_optimized)(
            i,
            extended_noised_mesh,
            output_directory,
            gt_normals,
            filter_iterations,
            shared_adjacency_graph  # PARAMETRO CHIAVE PER L'OTTIMIZZAZIONE
        )
        for i in range(num_faces)
    )
    
    # Statistiche finali
    successful = sum(1 for r in results if r is not None)
    print(f"\nProcessing completato:")
    print(f"- Facce processate con successo: {successful}/{num_faces}")
    print(f"- Tasso di successo: {successful/num_faces*100:.1f}%")
    
    if successful < num_faces:
        failed = [i for i, r in enumerate(results) if r is None]
        print(f"- Facce fallite: {failed[:10]}{'...' if len(failed) > 10 else ''}")

if __name__ == "__main__":
    main()