import AdjacencyGraph
import time
import numpy as np
    
def find_second_ring_neighbor_optimized(face_index, adjacency_graph):
    """
    Optimized version of function to find second ring neighbor

    Args:
        face_index: index of face.
        adjacency_graph: instance of AdjacencyGraph
    
    Returns:
        set: neighbors of second ring.
    """
    return adjacency_graph.get_second_ring_neighbors(face_index)

def calculate_alpha_g_optimized(filtered_normals, adjacency_graph):
    """
    Optimized computing of alpha_g using adjacency graph optimized.

    Args:
        filtered_normals: filtered normals.
        adjacency_graph: instance of AdjacencyGraph
    
    Returns:
        set: set of neighbors of second ring.
    """
    num_faces = len(filtered_normals)
    alpha_g_values = np.zeros(num_faces)

    norms = np.linalg.norm(filtered_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_normals = filtered_normals / norms

    all_face_indices = list(range(num_faces))
    second_ring_dict = adjacency_graph.batch_get_second_ring_neighbors(all_face_indices)

    for face_index in range(num_faces):
        second_ring = second_ring_dict[face_index]

        if len(second_ring) < 2:
            alpha_g_values[face_index] = 0.0
            continue
        
        second_ring_list = list(second_ring)
        second_ring_normals = normalized_normals[second_ring_list]

        n_faces_ring = len(second_ring_list)

        dot_products = np.dot(second_ring_normals, second_ring_normals.T)

        np.clip(dot_products, -1.0, 1.0, out=dot_products)

        angles = np.arccos(dot_products)

        upper_triangle_mask = np.triu(np.ones((n_faces_ring, n_faces_ring), dtype=bool), k=1)

        if np.any(upper_triangle_mask):
            alpha_g_values[face_index] = np.max(angles[upper_triangle_mask])
        else:
            alpha_g_values[face_index] = 0.0
    
    return alpha_g_values

class PatchData: 
    def __init__(self, expanded_mesh, face_index, gt_normals, filter_num_iterations, shared_adjacency_graph=None):
        """
        Class to compute datas of patch of given face.

        Args:
            expanded_mesh: instance of Expanded Mesh of selected one
            face_index: index of face of which compute patch.
            gt_normals: normals of patch ground truth.
            filter_num_iterations: number of iterations of mesh filtering to compute alpha_n and alpha_g
            share_adjacency_graph: shared instance of AdjacencyGraph to avoid re-computing.
        """
        self.mesh_vertices_origin = expanded_mesh.vertices
        self.mesh_faces = expanded_mesh.faces
        self.face_index = face_index
        self.gt_normals = gt_normals

        if shared_adjacency_graph is not None:
            self.adjacency_graph = shared_adjacency_graph
        else:
            self.adjacency_graph = AdjacencyGraph(expanded_mesh.face_adjacency)
        
        self.vertices = np.copy(self.mesh_vertices_origin)
        self.faces = self.mesh_faces
        self.centroids = expanded_mesh.centroids
        self.normals = expanded_mesh.normals

        gt_norms = np.linalg.norm(self.gt_normals, axis=1, keepdims=True)
        gt_norms[gt_norms == 0] = 1
        self.gt_normals = self.gt_normals / gt_norms

        self.areas = expanded_mesh.areas

        filtered_normals = self.apply_normal_filtering_optimized(filter_num_iterations)

        self.alpha_n_values = self.calculate_alpha_n(self.normals, filtered_normals)

        self.alpha_g_values = calculate_alpha_g_optimized(filtered_normals, self.adjacency_graph)

        self.patch_faces = self.calc_patch_optimized()
        self.adjacency_matrix = self.build_patch_adjacency_matrix_optimized()

        self.align_patch()
        self.features = self.calculate_features_optimized()

    def apply_normal_filtering_optimized(self, num_iterations=1):
        """Normals filtering optimized"""
        filtered_normals = self.normals.copy()

        for iteration in range(num_iterations):
            new_normals = filtered_normals.copy()
            for face_idx in range(len(self.normals)):
                adjacent_faces = list(self.adjacency_graph.get_first_ring_neighbors(face_idx))
                if len(adjacent_faces) > 0:
                    avg_normal = np.mean(filtered_normals[adjacent_faces], axis=0)
                    norm = np.linalg.norm(avg_normal)
                    if norm > 0:
                        new_normals[face_idx] = avg_normal / norm
            
            filtered_normals = new_normals
    
        return filtered_normals

    def calc_patch_optimized(self):
        """Patch computing optimized"""
        central_face = self.faces[self.face_index]
        central_face_centroid = np.mean(self.vertices[central_face], axis=0)

        second_ring_neighbors = self.adjacency_graph.get_second_ring_neighbors(self.face_index)

        if len(second_ring_neighbors) > 0:
            areas = []
            for neighbor_idx in second_ring_neighbors:
                face = self.faces[neighbor_idx]
                normal = np.cross(
                    self.vertices[face[1]] - self.vertices[face[0]],
                    self.vertices[face[2]] - self.vertices[face[0]]
                )
                area = np.linalg.norm(normal)/2
                areas.append(area)
            a_i = np.mean(areas)
        else:
            face = self.faces[self.face_index]
            normal = np.cross(
                    self.vertices[face[1]] - self.vertices[face[0]],
                    self.vertices[face[2]] - self.vertices[face[0]]
            )
            a_i = np.linalg.norm(normal) / 2
        
        k = 4
        radius = k * np.sqrt(a_i)

        face_vertices = self.vertices[self.faces]
        distances = np.linalg.norm(
            face_vertices - central_face_centroid[None, None, :], axis=2
        )

        faces_in_patch = np.any(distances < radius, axis=1)
        return set(np.where(faces_in_patch)[0])

    def build_patch_adjacency_matrix_optimized(self):
        """Build adjacency matrix optimized """
        patch_face_list = sorted(list(self.patch_faces))
        num_patch_faces = len(patch_face_list)
        patch_adj = np.zeros((num_patch_faces, num_patch_faces))

        global_to_local = {
            global_idx: local_idx for local_idx, global_idx in enumerate(patch_face_list)
        }

        for local_idx, global_face in enumerate(patch_face_list):
            neighbors = self.adjacency_graph.get_first_ring_neighbors(global_face)

            for neighbor in neighbors:
                if neighbor in global_to_local:
                    neighbor_local = global_to_local[neighbor]
                    patch_adj[local_idx, neighbor_local] = 1
                    patch_adj[neighbor_local, local_idx] = 1
        
        return patch_adj

    def calculate_features_optimized(self):
        """Optimized features computing"""
        patch_faces_list = list(self.patch_faces)

        min_area, max_area = np.min(self.areas), np.max(self.areas)
        area_range = max_area - min_area if max_area != min_area else 1.0

        patch_alpha_n = self.alpha_n_values[patch_faces_list]
        patch_alpha_g = self.alpha_g_values[patch_faces_list]

        min_alpha_n, max_alpha_n = np.min(patch_alpha_n), np.max(patch_alpha_n)
        alpha_n_range = max_alpha_n - min_alpha_n if max_alpha_n != min_alpha_n else 1.0
        
        min_alpha_g, max_alpha_g = np.min(patch_alpha_g), np.max(patch_alpha_g)
        alpha_g_range = max_alpha_g - min_alpha_g if max_alpha_g != min_alpha_g else 1.0
        
        features = []

        for face in patch_faces_list:
            # Normalizzazioni vettoriali
            centroid_norm = (self.centroids[face] + 1) / 2
            normal_norm = (self.normals[face] + 1) / 2
            area_norm = (self.areas[face] - min_area) / area_range
            alpha_n_norm = (self.alpha_n_values[face] - min_alpha_n) / alpha_n_range
            alpha_g_norm = (self.alpha_g_values[face] - min_alpha_g) / alpha_g_range
            vertices_norm = (self.vertices[self.faces[face]] + 1) / 2
            
            feature_vector = np.hstack([
                centroid_norm, normal_norm, area_norm, alpha_n_norm, alpha_g_norm, vertices_norm.flatten()
            ])
            features.append(feature_vector)
        
        return np.array(features).T

    def calculate_alpha_n(self, original_normals, filtered_normals):
        """Calcola alpha^n vettorialmente"""
        orig_norms = np.linalg.norm(original_normals, axis=1, keepdims=True)
        orig_norms[orig_norms == 0] = 1
        original_normalized = original_normals / orig_norms
        
        filt_norms = np.linalg.norm(filtered_normals, axis=1, keepdims=True)
        filt_norms[filt_norms == 0] = 1
        filtered_normalized = filtered_normals / filt_norms
        
        dot_products = np.sum(original_normalized * filtered_normalized, axis=1)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        
        return np.arccos(dot_products)
    
    def align_patch(self):
        # create an array of indices
        patch_face_indices = np.array(self.faces[list(self.patch_faces)])
        # Create copy of data related to patch
        patch_vertices = np.copy(self.vertices)
        patch_centroids = np.copy(self.centroids)
        patch_normals = np.copy(self.normals)
        # move patch to the origin 
        central_face_centroid = patch_centroids[self.face_index]
        patch_vertices[patch_face_indices] -= central_face_centroid
        patch_centroids -= central_face_centroid 
        
        # scaling vertices in unit bounding box
        selected_vertices = patch_vertices[patch_face_indices]
        min_coords = np.min(selected_vertices, axis=0)
        if selected_vertices.shape[0] > 0:
            min_coords = np.min(selected_vertices, axis=0)
            max_coords = np.max(selected_vertices, axis=0)
            bounding_box_size = np.max(max_coords-min_coords)
            if bounding_box_size > 0:
                scale_factor = 1.0 / bounding_box_size
                patch_vertices /= bounding_box_size
                patch_centroids *= scale_factor
        
        # Compute areas on scaled vertices 
        self.areas = self.areas * (scale_factor**2)
        
        # computze mu
        areas = self.areas[list(self.patch_faces)]
        a_m = np.max(areas)
        centroids = patch_centroids
        c_i = patch_centroids[self.face_index]
        distances = np.linalg.norm(centroids-c_i)
        sigma = np.median(distances)
        if(sigma == 0) : sigma = 0.1
        mu = (areas/a_m)*np.exp(-distances*3)
        
        # compute n_j'
        v = centroids - c_i
        normals = patch_normals
        u = np.cross(v, normals)
        w = np.cross(u,v)
        w = w / np.linalg.norm(w)
        n_first = 2*(normals*w)*w - normals
        
        #compute T_i
        T_i = np.zeros((3,3))
        for j in range(len(mu)):
            T_i += mu[j] * np.outer(n_first[j], n_first[j])
        
        # compute eigenvalues and eigenvectors
        try: 
            eigenvalues, eigenvectors = np.linalg.eigh(T_i)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, sorted_indices]
            R_i = eigenvectors
        except np.linalg.LinAlgError:
            print(f"Warning: Eigendecomposition failed for face {self.face_index}. Using identity matrix")
            R_i = np.identity(3)

        # Multiply vertex features for R (rotation matrix)
        patch_vertices = (R_i.T @ patch_vertices.T).T
        patch_centroids = (R_i.T @ patch_centroids.T).T
        patch_normals = (R_i.T @ patch_normals.T).T
        
        self.vertices = patch_vertices
        self.centroids = patch_centroids
        self.normals = patch_normals
    
    def save_to_mat(self, filename):
        """Salva i dati in formato .mat"""
        import scipy.io
        data = {
            "MAT": self.adjacency_matrix,
            "FEA": self.features,
            "GT": ((self.gt_normals[self.face_index]+1)/2).T.reshape(3,1),
            "NOR": ((self.normals[self.face_index]+1)/2).T.reshape(3,1)
        }
        scipy.io.savemat(filename, data)

# Funzione di processing ottimizzata per il parallelismo
def process_patch_optimized(face_index, expanded_mesh, output_dir, gt_normals, 
                          num_filter_iterations, shared_adjacency_graph):
    """
    Versione ottimizzata della funzione di processing che usa il grafo condiviso
    """
    try:
        patch = PatchData(
            expanded_mesh, face_index, gt_normals, 
            num_filter_iterations, shared_adjacency_graph
        )
        filename = f"{output_dir}/0_{face_index}.mat"
        patch.save_to_mat(filename)
        print(f"Face {face_index} patch correctly saved.")
        return filename
    except Exception as e:
        print(f"Errore durante l'elaborazione della faccia {face_index}: {e}")
        import traceback
        traceback.print_exc()
        return None