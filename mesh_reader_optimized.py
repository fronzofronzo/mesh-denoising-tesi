import trimesh
import numpy as np
from scipy.sparse import coo_matrix,lil_matrix
import scipy.io
import os, argparse
from joblib import Parallel, delayed
from generate_noise import add_gaussian_noise_paper
from ExpandedMesh import ExpandedMesh 

#Function to calculate centroid given faces and vertices
def calc_centroid(face, vertices):
    centroids = np.mean(vertices[face], axis=0)
    return centroids

#Calculate normal of faces
def calc_normal(face, vertices):
    p1 = vertices[face[0]] # gets firts vertex of each face
    p2 = vertices[face[1]]
    p3 = vertices[face[2]]
    
    v = p2-p1
    w = p3-p1
    
    normals = np.cross(v,w)
    
    return normals

# calculate the area of faces
def calc_area(face, vertices):
    normal = calc_normal(face, vertices)
    area = np.linalg.norm(normal)/2
    return area

class PatchData:
    def __init__(self, expanded_mesh, face_index, gt_normals):
        self.mesh_vertices_origin = expanded_mesh.vertices
        self.mesh_faces = expanded_mesh.faces
        self.face_index = face_index
        self.gt_normals = gt_normals
        
        # creates copies
        self.vertices = np.copy(self.mesh_vertices_origin)
        self.faces = self.mesh_faces
        
        # sets patch properties
        self.centroids = expanded_mesh.centroids
        self.normals =  expanded_mesh.normals
        
        gt_norms = np.linalg.norm(self.gt_normals, axis=1, keepdims=True)
        gt_norms[gt_norms == 0] = 1
        self.gt_normals = self.gt_normals / gt_norms
        
        self.areas =  expanded_mesh.areas

        self.adj_graph =  expanded_mesh.face_adjacency
        # Dentro PatchData.__init__
        self.adj_pairs = None 
        try:
            self.adj_pairs = self.adj_graph
            if self.adj_pairs is not None:
                if self.adj_pairs.size > 0:
                    face_to_check = self.face_index # Usa l'indice corrente
                    pairs_with_current_face = self.adj_pairs[(self.adj_pairs[:, 0] == face_to_check) | (self.adj_pairs[:, 1] == face_to_check)]
        except Exception as e:
            print(f"Error obtaining face adjacency from trimesh: {e}")
            self.adj_pairs = np.empty((0, 2), dtype=int)

        self.patch_faces = self.calc_patch(self.faces[face_index], self.faces, self.vertices)
        self.adjacency_matrix = self.build_patch_adjacency_matrix()

        self.align_patch()
        self.features = self.calculate_features()
        
    def build_patch_adjacency_matrix(self):
        patch_face_list = sorted(list(self.patch_faces)) 
        map_global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(patch_face_list)}
        num_patch_faces = len(patch_face_list)
        patch_adj = np.zeros((num_patch_faces, num_patch_faces))
        adj_pairs = self.adj_pairs
        if adj_pairs is not None:
                # Filtra le coppie che contengono solo facce della patch
                patch_adj_pairs = adj_pairs[np.isin(adj_pairs[:, 0], patch_face_list) & 
                                        np.isin(adj_pairs[:, 1], patch_face_list)]
                
                # Costruisci la matrice locale
                for pair in patch_adj_pairs:
                    f1_global, f2_global = pair
                    # Mappa a indici locali (0..k-1)
                    f1_local = map_global_to_local[f1_global]
                    f2_local = map_global_to_local[f2_global]
                    # Imposta adiacenza
                    patch_adj[f1_local, f2_local] = 1
                    patch_adj[f2_local, f1_local] = 1 # Assicura simmetria
        
        return patch_adj
    
    
    
    # calculate patch given the central face
    def calc_patch(self, central_face, faces, vertices):
        current_face_index = np.where([set(face) == set(central_face) for face in faces])[0][0]
        central_face_centroid = calc_centroid(central_face, vertices)
        adj_pairs = self.adj_graph
        
        first_ring_neighbors_indices = np.array([], dtype=int) # Default

        if adj_pairs is not None and adj_pairs.size > 0:
             try:
                 neighbors1 = adj_pairs[adj_pairs[:, 0] == current_face_index, 1] 
                 neighbors2 = adj_pairs[adj_pairs[:, 1] == current_face_index, 0]
                 first_ring_neighbors_indices = np.unique(np.concatenate((neighbors1, neighbors2)))
             except IndexError as e:
                  print(f"Errore di indicizzazione cercando vicini per faccia {current_face_index}: {e}")
             except Exception as e:
                  print(f"Errore generico cercando vicini per faccia {current_face_index}: {e}")
        else:
             print("Warning: Array adj_pairs non disponibile o vuoto.")

        # Calcolo secondo anello usando adj_pairs
        second_ring_neighbors = set()
        if adj_pairs is not None and adj_pairs.size > 0:
             for index in first_ring_neighbors_indices:
                 try:
                      n1 = adj_pairs[adj_pairs[:, 0] == index, 1]
                      n2 = adj_pairs[adj_pairs[:, 1] == index, 0]
                      neighbors_of_index = np.unique(np.concatenate((n1, n2)))
                      second_ring_neighbors.update(neighbors_of_index)
                 except IndexError as e:
                      print(f"Index error finding neighbors {index}: {e}")
                 except Exception as e:
                      print(f"Generic error searching neighbors for index {index}: {e}")

             second_ring_neighbors -= set(first_ring_neighbors_indices)
             second_ring_neighbors.discard(current_face_index)
        
        areas = np.array([calc_area(faces[i], vertices) for i in second_ring_neighbors])
        a_i = np.mean(areas)
        k = 4
        self.radius = k * np.sqrt(a_i)
        patch = set()
        for i, face in enumerate(faces):
            ver = vertices[face]
            if np.any(np.linalg.norm(ver - central_face_centroid, axis=1) < self.radius):
                patch.add(i)
        return patch

    
    # method to align patch with normal tensor voting
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
        
    # calculate the features of the mesh
    def calculate_features(self):
        min_area = np.min(self.areas)
        max_area = np.max(self.areas) 
        features=[]
        for face in self.patch_faces:
            #normalize from [-1;1] to [0;1]
            centroid = (self.centroids[face] + 1)/2
            normal = (self.normals[face] +1)/2 
            area = self.areas[face]
            area = (area-min_area)/(max_area-min_area)
            # neighbors normalization
            neighbors = np.count_nonzero(self.adj_pairs[:, 0] == face) + np.count_nonzero(self.adj_pairs[:, 1] == face)
            neighbors = neighbors/ 3 * 0.5
            vertices = (self.vertices[self.faces[face]] +1)/2
            features.append(np.hstack([centroid, normal, area, neighbors, vertices.flatten()]))
        return np.array(features).T
            
    def save_to_mat(self, filename):
        data = {
            "MAT": self.adjacency_matrix,
            "FEA": self.features,
            "GT": ((self.gt_normals[self.face_index]+1)/2).T.reshape(3,1),
            "NOR": ((self.normals[self.face_index]+1)/2).T.reshape(3,1)
        }
        scipy.io.savemat(filename, data)

#wrap function to process patches
def process_patch(face_index, expanded_mesh, output_dir, gt_normals):
    try:
        patch = PatchData(expanded_mesh, face_index, gt_normals)
        filename = os.path.join(output_dir, f"0_{face_index}.mat")
        patch.save_to_mat(filename)
        print(f"Face {face_index} patch correctly saved.")
        return filename
    except Exception as e:
        print(f"Errore durante l'elaborazione della faccia {face_index}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Generate patches for selected mesh")
    parser.add_argument('mesh_name', default=None,
                        help="Mesh to which generate noise",
                        type=str)
    parser.add_argument('noise_level', default="0.1",
                        help="Select noise level for mesh",
                        type=float)
    args = parser.parse_args()
    mesh_name = args.mesh_name
    noise_level = args.noise_level
    model_path = os.path.join(script_dir, "testing_models", f"{mesh_name}_gt.obj")
    noised_model_path = os.path.join(script_dir, "testing_models", f"{mesh_name}_noised_{noise_level}_Gaussian.obj")
    noised_mesh = trimesh.load_mesh(noised_model_path)
    mesh = trimesh.load_mesh(model_path)
    gt_normals = []
    for face,i in enumerate(mesh.faces):
        gt_normals.append(calc_normal(i, mesh.vertices))
    gt_normals = np.array(gt_normals)
    extended_noised_mesh = ExpandedMesh(noised_mesh)
    num_faces = len(noised_mesh.faces)
    os.makedirs(f"samples/{mesh_name}_{noise_level}")
    output_directory = os.path.join(script_dir, "testing_samples", f"{mesh_name}_{noise_level}")
    # Process patch to mesh
    Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(process_patch)(
            i,
            extended_noised_mesh,
            output_directory,
            gt_normals
        )
        for i in range(num_faces)
    )