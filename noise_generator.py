import trimesh
import numpy as np
from scipy.stats import t, chi2
from typing import List, Tuple, Union
import random, os, argparse
import ExpandedMesh
import ExpandedMesh

class NoiseParams:
    def __init__(self, noise_level: float = 0.1, impulsive_level: float = 0.1, scale: float = 1.0):
        self.noise_level = noise_level
        self.impulsive_level = impulsive_level
        self.scale = scale

class NoiseDirection:
    NORMAL = 0
    RANDOM = 1

class MeshProcessor:
    def __init__(self, mesh, seed=42):
        self.mesh = mesh
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def _calculate_average_edge_length(self):
        """Calculate average length of mesh's edges"""
        edge_lengths = self.mesh.edges_unique_length
        return np.mean(edge_lengths)
    
    def _generate_random_directions(self, count):
        """
        Generate random normalized directions
        
        Params:
            count (int): directions to be generated.

        Returns: 
            float : directions normalized
        """
        directions = np.random.randn(count, 3)
        norms = np.linalg.norm(directions, axis=1)
        return directions / norms[:, np.newaxis]
    
    def _generate_impulsive_noise(self, num_vertices, impulsive_level,
                                   standard_deviation):
        """
        Generate impulsive noise for some vertices selected randomly
        
        Args:
            num_vertices (int): number of vertices of the mesh.
            impulsive_level (float): percentage of vertices that will be modified by noise.
            standard_deviation (float): determinate the variation of values that'll be generated.

        Returns:
            tuple (list(int), list(float)): list of selected indices of vertex to move and 
            values of displacement randomly generated.
        """
        impulsive_vertex_count = int(num_vertices * impulsive_level)
        selected_indices = np.random.choice(num_vertices, impulsive_vertex_count, replace=False)
        gaussian_values = np.random.normal(0, standard_deviation, impulsive_vertex_count)
        return selected_indices.tolist(), gaussian_values.tolist()
    
    def _get_noise_numbers(self, noise_type, average_length, noise_params):
        """
        Generate noise numbers for selected type.

        Args:
            noise_type (str): noise type to be applied. It must be one of: gaussian, student, chisquared
            and uniform. If none of the above is selected, then it's applied gaussian by default.
            average_length (float): average length of mesh edges.
            noise_params (NoiseParams): parameters of noise level. 
        
        Returns:
            float: noise numbers of specified type. 
        """
        num_vertices = len(self.mesh.vertices)
        if noise_type.lower() == "gaussian":
            return np.random.normal(0, average_length * noise_params.noise_level, num_vertices)
        elif noise_type.lower() == "student":
            df = 2
            return t.rvs(df, size=num_vertices) * average_length * noise_params.noise_level
        elif noise_type.lower() == "chisquared":
            df = 2
            return chi2.rvs(df, size=num_vertices) * average_length * noise_params.noise_level
        elif noise_type.lower() == "uniform":
            return np.random.uniform(-1,1,num_vertices) * average_length * noise_params.noise_level
        else:
            return np.random.normal(0, average_length * noise_params.noise_level, num_vertices)
        
    def generate_noise(self, noise_params, noise_direction, noise_type):
        """
        Generate noise on 3D mesh

        Args:
            noise_params: parameters of noise.
            noise_direction: 0 for normal direction, 1 for casual direction.
            noise_type: noise type ("Impulsive", "Student", "ChiSquared", "Gaussian", etc.)

        Returns:
            trimesh.Trimesh: noised mesh.
        """
        noisy_mesh = self.mesh.copy()

        num_vertices = len(noisy_mesh.vertices)
        if num_vertices == 0:
            return noisy_mesh

        average_length = self._calculate_average_edge_length()

        if not hasattr(noisy_mesh.visual, "vertex_normals") or noisy_mesh.vertex_normlas is None:
            noisy_mesh.vertex_normals

        random_directions = None
        if noise_direction == NoiseDirection.RANDOM:
            if noise_type.lower() == "impulsive":
                impulsive_vertex_count = int(num_vertices * noise_params.impulsive_level)
                random_directions = self._generate_random_directions(impulsive_vertex_count)
            else: 
                random_directions = self._generate_random_directions(num_vertices)
        
        if noise_type.lower() == "impulsive":
            standard_deviation = average_length * noise_params.noise_level
            selected_indices, gaussian_values = self._generate_impulsive_noise(
                num_vertices, noise_params.impulsive_level, standard_deviation
            )

            if noise_direction == NoiseDirection.NORMAL:
                for i,idx in enumerate(selected_indices):
                    vertex_normal = noisy_mesh.vertex_normals[idx]
                    displacement = vertex_normal * gaussian_values[i]
                    noisy_mesh.vertices[idx] += displacement

            elif noise_direction == NoiseDirection.RANDOM:
                for i, idx in enumerate(selected_indices):
                    displacement = random_directions[i] * gaussian_values[i]
                    noisy_mesh.vertices[idx] += displacement

        elif noise_type.lower() in ["student", "chisquared"]:
            noise_numbers = self._get_noise_numbers(noise_type, average_length, noise_params)
            
            if noise_direction == NoiseDirection.NORMAL:
                for i in range(num_vertices):
                    vertex_normal = noisy_mesh.vertex_normals[i]
                    displacement = (
                        average_length * noise_params.scale * vertex_normal * noise_numbers[i]
                    )
                    noisy_mesh.vertices[i] += displacement
            elif noise_direction == NoiseDirection.RANDOM:
                for i in range(num_vertices):
                    displacement = (
                        average_length * noise_params.scale 
                        * random_directions[i] * noise_numbers[i]
                    )
                    noisy_mesh.vertices[i] += displacement
        
        else:
            noise_numbers = self._get_noise_numbers(noise_type, average_length, noise_params)

            if noise_direction == NoiseDirection.NORMAL:
                for i in range(num_vertices):
                    vertex_normal = noisy_mesh.vertex_normals[i]
                    displacement = vertex_normal * noise_numbers[i]
                    noisy_mesh.vertices[i] += displacement
            elif noise_direction == NoiseDirection.RANDOM:
                for i in range(num_vertices):
                    displacement = random_directions[i] * noise_numbers[i]
                    noisy_mesh.vertices[i] += displacement
        
        noisy_mesh._cache.clear()

        return noisy_mesh

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Generate noise of selected level for a selected mesh")
    parser.add_argument('mesh_name', default=None,
                        help="Mesh to which generate noise",
                        type=str)
    parser.add_argument('noise_level', default="0.1",
                        help="Select noise level for mesh",
                        type=float)
    parser.add_argument('impulsive_level', default=None,
                        help="Select impulsive level for noise",
                        type=float)
    args = parser.parse_args()
    mesh_name = args.mesh_name
    noise_level = args.noise_level
    impulsive_level = args.impulsive_level
    model_path = os.path.join(script_dir, "testing_models", mesh_name + "_gt.obj")
    mesh = trimesh.load_mesh(model_path)

    processor = MeshProcessor(mesh)

    noise_params = NoiseParams(noise_level, impulsive_level, scale=1.0)
            
    noisy_mesh = processor.generate_noise(
        noise_params,
        NoiseDirection.NORMAL,
        "Gaussian"
    )
    noised_mesh_path = os.path.join(script_dir, "testing_models", f"{mesh_name}_noised_{noise_level}_Gaussian.obj" )
    noisy_mesh.export(file_obj=noised_mesh_path)
