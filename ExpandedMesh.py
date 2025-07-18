import trimesh
import numpy as np
class ExpandedMesh:
    """
    This class presents an expansion of a classic mesh structure. It computes also centroids, normals and
    areas of faces of mesh.
    """
    def __init__(self, mesh):
        self.vertices = mesh.vertices
        self.faces = mesh.faces

        # compute normals 
        p1 = self.vertices[self.faces[:, 0]]
        p2 = self.vertices[self.faces[:, 1]]
        p3 = self.vertices[self.faces[:, 2]]

        v = p2-p1
        w = p3-p1

        unnormalized_normals = np.cross(v,w)

        # compute areas
        self.areas = np.linalg.norm(unnormalized_normals, axis=1) / 2.0

        # normalize norms
        norms = np.linalg.norm(unnormalized_normals, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        self.normals = unnormalized_normals / norms

        #compute centroids
        self.centroids = np.mean(self.vertices[self.faces], axis=1)

        self.face_adjacency = mesh.face_adjacency