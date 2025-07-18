
import trimesh
import numpy as np
from scipy.sparse import lil_matrix
import scipy.io

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
    def __init__(self, path, face_index):
        self.mesh = trimesh.load_mesh(path)
        self.face_index = face_index
        
        self.vertices = self.mesh.vertices
        
        self.faces = self.mesh.faces
        self.centroids = self.mesh.triangles_center
        self.normals = self.mesh.face_normals
        self.areas = self.mesh.area_faces
        self.mesh_adjacency = self.cell_adjacency_matrix(self.faces)
        
        self.patch_faces = self.calc_patch(self.faces[face_index], self.faces, self.vertices)
        self.adjacency_matrix = self.build_patch_adjacency_matrix()
        self.align_patch()
        self.features = self.calculate_features()
    
    # Build the adjacency matrix of the entire mesh
    def cell_adjacency_matrix(self, faces):
        n_faces = len(faces)
        A = lil_matrix((n_faces, n_faces), dtype=int)
        # maps each vertex to faces that contains them
        vertex_to_faces = {}
        for i,face in enumerate(faces):
            for v in face:
                if v not in vertex_to_faces:
                    vertex_to_faces[v] = set()
                vertex_to_faces[v].add(i)
                
        #build adjacency matrix 
        for i, face1 in enumerate(faces):
            for v in face1:
                for neighbor_face in vertex_to_faces[v]:
                    if neighbor_face != i:
                        A[i,neighbor_face] = 1 
        return A
    
    # calculate patch given the central face
    def calc_patch(self, central_face, faces, vertices):
        face_index = np.where([set(face) == set(central_face) for face in faces])[0][0]
        central_face_centroid = calc_centroid(central_face, vertices)
        #calculate adjacency matrix
        A = self.mesh_adjacency
        # find first ring neighbors 
        first_ring_neighbors_index =  A.getrow(face_index).nonzero()[1]
        #find second ring neighbors
        second_ring_neighbors = set()
        for index in first_ring_neighbors_index:
            second_ring_neighbors.update(A.getrow(index).nonzero()[1])
        second_ring_neighbors -= set(first_ring_neighbors_index)
        second_ring_neighbors.discard(face_index)
        # calculate ray to find the patch
        areas = np.array([calc_area(faces[i], vertices) for i in second_ring_neighbors])
        a_i = np.mean(areas)
        k = 4
        self.radius = k * np.sqrt(a_i)
        patch = set()
        for i, face in enumerate(faces):
            ver = vertices[face]
            #print(np.linalg.norm(ver-central_face_centroid, axis=1))
            if np.any(np.linalg.norm(ver - central_face_centroid, axis=1) < self.radius):
                patch.add(i)
        return patch

    # build the adjacency matrix of the patch
    def build_patch_adjacency_matrix(self):
        num_patch_faces = len(self.patch_faces)
        matrix = np.zeros((num_patch_faces, num_patch_faces), dtype=np.uint8)
        face_vertex_sets = [set(self.faces[face]) for face in self.patch_faces]
        
        for i, vert_i in enumerate(face_vertex_sets):
            for j, vert_j in enumerate(face_vertex_sets):
                if i != j and len(vert_i & vert_j) >= 2:
                    matrix[i,j] = 1
        return matrix
    
    # method to align patch with normal tensor voting
    def align_patch(self):
        # move patch to the origin 
        central_face_centroid = self.centroids[self.face_index]
        self.vertices[self.faces[list(self.patch_faces)]] -= central_face_centroid
        cent = self.centroids - central_face_centroid
        self.centroids = cent
        
        # scaling vertices in unit bounding box
        selected_vertices = self.vertices[self.faces[list(self.patch_faces)]].reshape(-1,3)
        min_coords = np.min(selected_vertices, axis=0)
        max_coords = np.max(selected_vertices, axis=0)
        bounding_box_size = np.max(max_coords-min_coords)
        if bounding_box_size > 0:
            self.vertices[self.faces[list(self.patch_faces)]] /= bounding_box_size
        
        # compute mu_j
        areas = self.areas[list(self.patch_faces)]
        a_m = np.max(areas)
        centroids = self.centroids[list(self.patch_faces)]
        c_i = self.centroids[self.face_index]
        distances = np.linalg.norm(centroids-c_i)
        sigma = np.median(distances)
        print("a_m: ", a_m)
        print("sigma: ", sigma)
        mu = (areas/a_m)*np.exp(-distances/sigma)
        
        # compute n_j'
        v = centroids - c_i
        normals = self.normals[list(self.patch_faces)]
        u = np.cross(v, normals)
        w = np.cross(u,v)
        w = w / np.linalg.norm(w)
        n_first = 2*(normals*w)*w - normals
        
        #compute T_i
        T_i = np.zeros((3,3))
        for j in range(len(mu)):
            T_i += mu[j] * np.outer(n_first[j], n_first[j])
        
        # compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(T_i)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        R_i = eigenvectors
        self.centroids = (R_i.T @ self.centroids.T).T
        self.normals = (R_i.T @ self.normals.T).T
    
    # calculate the features of the mesh
    def calculate_features(self): 
        features=[]
        for face in self.patch_faces:
            centroid = self.centroids[face]
            centroid = (centroid + 1)/2
            normal = self.normals[face]
            normal = (normal + 1)/2
            area = self.areas[face]
            area = area/self.radius**2
            neighbors = self.mesh_adjacency[face].getnnz()
            neighbors = (neighbors -12)/6 * 0.5 + 0.5
            vertices = self.vertices[self.faces[face]]
            vertices = (vertices+1)/2
            features.append(np.hstack([centroid, normal, area, neighbors, vertices.flatten()]))
        return np.array(features).T
            
    def save_to_mat(self, filename):
        data = {
            "MAT": self.adjacency_matrix,
            "FEA": self.features,
            "GT": self.normals[self.face_index].T,
            "NOR": np.array([0,0,0]).T
        }
        scipy.io.savemat(filename, data)

mesh = trimesh.load_mesh("armadillo_gaus_n3.obj")
num_faces = len(mesh.faces)
for i in range(num_faces):
    patch = PatchData("armadillo_gaus_n3.obj", i)
    path = "samples/0_" + str(i) + ".mat"
    patch.save_to_mat(path)