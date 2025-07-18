from mesh_reader_optimized import PatchData

import numpy as np
import h5py
import os
import trimesh

def surface_denoising(self, mesh_name):
    mesh_path = os.path.join("models", f"{mesh_name}.obj")
    mesh = trimesh.load_mesh(mesh_path)
    num_faces = len(mesh.faces)
    for i in range(1):
        mat_path = os.path.join("samples", mesh_name, f"0_{i}.mat")
        f = h5py.File(mat_path)
        data = f.get('data')
        data = np.array(data)
        print(data)
    return

if __name__ == "__main__":
    surface_denoising("girl")