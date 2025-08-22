import trimesh
import numpy as np

class FaceNeighborType:
    EDGE_BASED = 0
    VERTEX_BASED = 1
    RADIUS_BASED = 2

def getAverageEdgeLength(mesh):
    return mesh.edges_unique_length.mean()

def getFaceArea(mesh):
    faces = mesh.faces
    areas = []
    for face in faces:
        v1 = mesh.vertices[face[0]]
        v2 = mesh.vertices[face[1]]
        v3 = mesh.vertices[face[2]]

        e1 = v2 - v1
        e2 = v1 - v3
        area = np.linalg.norm(np.cross(e1,e2)) * 0.5
        areas.append(area)
    return np.array(areas)

def getFaceCentroid(mesh):
    return np.mean(mesh.vertices[mesh.faces], axis=1)

def getFaceNormals(mesh):
    v1 = mesh.vertices[mesh.faces[:, 0]]
    v2 = mesh.vertices[mesh.faces[:, 1]]
    v3 = mesh.vertices[mesh.faces[:, 2]]

    u = v2-v1
    w = v3-v1

    return np.cross(u,w)

def getFaceNeighbor(mesh, face, face_neighbor_type): 
    if(face_neighbor_type == FaceNeighborType.EDGE_BASED):
        boolean_matrix = np.isin(mesh.faces, face)
        print(boolean_matrix)
        condition = np.sum(boolean_matrix, axis=1) == 2
        indices = np.where(condition)
        return indices[0]
    elif(face_neighbor_type == FaceNeighborType.VERTEX_BASED):
        boolean_matrix = np.isin(mesh.faces, face)
        neighbor_mask = np.any(boolean_matrix, axis=1)
        all_indices = np.where(neighbor_mask)[0]
        face_index = np.where(np.all(mesh.faces == face,axis=1))[0] 
        return all_indices[all_indices != face_index]

def getAllFaceNeighbor(mesh, face_neighbor_type):
    neighbors_matrix = []
    for face in mesh.faces: 
        neighbors_matrix.append(getFaceNeighbor(
            mesh, face, face_neighbor_type
        ))
    return np.array(neighbors_matrix)

def getBoundaryVertices(mesh):
    """
    Identify mesh border vertices.

    Args:
        mesh(ExpandedMesh or trimesh.Trimesh): target mesh.

    Returns:
        Set[int]: indices of border vertex.
    """
    try:
        if hasattr(mesh, 'face_adjacency'):
            face_adjacency = mesh.face_adjacency
        else:
            face_adjacency = []
        
        if hasattr(mesh, 'edges_unique_counts'):
            edge_counts = mesh.edges_unique_counts
            boundary_edge_indices = np.where(edge_counts == 1)[0]
            boundary_edges = mesh.edges_unique[boundary_edge_indices]
            return set(np.unique(boundary_edges.flatten()))
        
        edge_counts = {}
        for face in mesh.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1)%3]]))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        boundary_vertices = set()
        for edge, count in edge_counts.items():
            if count == 1:
                boundary_vertices.update(edge)
        
        return boundary_vertices
    except Exception as e:
        print(f"Error while searching boundary vertices: {e}")
        return set()

def updateVertexPosition(mesh,
                        filtered_normals,
                        iteration_number,
                        fixed_boundary, dumping_factor):
    new_vertices = np.zeros_like(mesh.vertices)
    boundary_vertices = getBoundaryVertices(mesh)
    damping_factor = dumping_factor
    for i in range(iteration_number):
        current_vertices = mesh.vertices.copy()
        new_vertices = mesh.vertices.copy()
        centroids = getFaceCentroid(mesh)
        for idx, vertex in enumerate(mesh.vertices):
            new_vertex = vertex
            if(fixed_boundary and (idx in boundary_vertices)):
                new_vertices[idx] = new_vertex
            else:
                face_num = 0.0
                temp_point = np.zeros((3,), dtype=float)
                vertex_faces = mesh.faces == idx
                vertex_faces = np.any(vertex_faces, axis=1)
                vertex_faces = np.where(vertex_faces)[0]
                for face in vertex_faces:
                    temp_normal = filtered_normals[face]
                    temp_centroid = centroids[face]
                    temp_point += np.dot(temp_normal, temp_centroid- new_vertex) * temp_normal 
                    face_num += 1.0
                if face_num > 0:
                    new_position = new_vertex + damping_factor * (temp_point/face_num)
                    new_vertices[idx] = new_position
        mesh.vertices = new_vertices
    return mesh.vertices


