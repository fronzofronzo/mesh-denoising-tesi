import numpy as np
import trimesh
import mesh_denoising_base as mdb
from mesh_denoising_base import FaceNeighborType
from collections import deque

def getVertexBasedFaceNeighbors(mesh, face):
    return mdb.getFaceNeighbor(
        mesh,
        face,
        FaceNeighborType.VERTEX_BASED)

'''def getRadiusBasedFaceNeighbor(mesh, face, radius):
    centroids = mdb.getFaceCentroid(mesh)
    face_idx = np.where(np.all(mesh.faces == face, axis=1))[0][0]
    print(f"Starting searching of neihbors radius based for face {face_idx}", flush=True)
    c_i = centroids[face_idx]
    visited = np.zeros_like(mesh.faces[:,0], dtype=bool)
    face_neighbor = []

    visited[face_idx] = True
    queue_face = deque()
    queue_face.append(face_idx)
    while(len(queue_face) != 0):
        print(f"Elements in queue: {len(queue_face)}", flush=True)
        temp_face = queue_face.popleft()
        if not temp_face == face_idx:
            face_neighbor.append(temp_face)
        temp_face_neighbors = getVertexBasedFaceNeighbors(mesh, mesh.faces[temp_face])
        for idx, neighbor in enumerate(temp_face_neighbors):
            temp_neighbor = mesh.faces[neighbor]
            temp_neighbor_idx = neighbor
            if(not visited[temp_neighbor_idx]):
                c_j = centroids[temp_neighbor_idx]
                distance = np.linalg.norm(c_i - c_j)
                if distance <= radius:
                    queue_face.append(temp_neighbor_idx)
                visited[temp_neighbor_idx] = True
    return np.array(face_neighbor)'''

def getRadiusBasedFaceNeighbor(mesh, face, radius):
    centroids = np.asarray(mdb.getFaceCentroid(mesh))

    face_array = np.asarray(mesh.faces)
    face_idx = np.where(np.all(face_array == face, axis=1))[0][0]
    print(f"Starting search of neighbors within radius for face {face_idx}", flush=True)
    c_i = centroids[face_idx]

    visited = np.zeros(len(face_array), dtype=bool)
    visited[face_idx] = True

    queue_face = deque([face_idx])
    face_neighbors = []

    face_neighbor_cache = {}

    while queue_face:
        current_face_idx = queue_face.popleft()
        if current_face_idx != face_idx:
            face_neighbors.append(current_face_idx)

        if current_face_idx not in face_neighbor_cache:
            neighbors = getVertexBasedFaceNeighbors(mesh, face_array[current_face_idx])
            face_neighbor_cache[current_face_idx] = neighbors
        else:
            neighbors = face_neighbor_cache[current_face_idx]
        
        for neighbor_idx in neighbors:
            if not visited[neighbor_idx]:
                c_j = centroids[neighbor_idx]
                distance = np.linalg.norm(c_i-c_j)
                if distance <= radius:
                    queue_face.append(neighbor_idx)
                visited[neighbor_idx] = True
    return np.array(face_neighbors)

def getAllFaceNeighborGMNF(
        mesh,
        face_neighbor_type,
        radius,
        include_central_face):
    print("Getting face neighbors", flush=True)
    all_face_neighbor = []
    for idx, face in enumerate(mesh.faces):
        print(f"Finding neighbor for face {idx}", flush=True)
        if face_neighbor_type == FaceNeighborType.VERTEX_BASED:
            face_neighbor = getVertexBasedFaceNeighbors(mesh, face)
        elif face_neighbor_type == FaceNeighborType.RADIUS_BASED:
            face_neighbor = getRadiusBasedFaceNeighbor(mesh,  face, radius)
        if include_central_face:
            face_neighbor = np.append(face_neighbor, idx)
        all_face_neighbor.append(face_neighbor)
    return all_face_neighbor

def gaussianWeight(distance, sigma):
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return np.exp(-0.5 * distance * distance / (sigma * sigma))

def normalDistance(n1,n2):
    return np.linalg.norm(n1 - n2)

"""def getRadius(multiple, mesh):
    centroid = mdb.getFaceCentroid(mesh)

    radius = 0.0
    num = 0.0
    for face_idx, face in enumerate(mesh.faces):
        print(f"Computing radius using {face_idx} centroid")
        fi = centroid[face_idx]
        for face_jdx, face_j in enumerate(mesh.faces):
            if face_idx != face_jdx:
                fj = centroid[face_jdx]
                radius += np.linalg.norm(fj - fi)
                num += 1
    return radius * multiple / num"""

def getRadius(multiple, mesh):
    centroid = mdb.getFaceCentroid(mesh)
    centroid = np.asarray(centroid)

    diff = centroid[:, np.newaxis, :] - centroid[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    triu_indices = np.triu_indices_from(dists, k=1)
    radius = np.sum(dists[triu_indices])
    num = len(triu_indices[0])
    return radius * multiple / num if num > 0 else 0.0

"""def getSigmaS(multiple, mesh):
    sigma_s = 0.0
    num = 0.0
    centroids = mdb.getFaceCentroid(mesh)
    for face_idx, face in enumerate(mesh.face):
        f_i = centroids[face_idx]
        for face_jdx, fface in enumerate(mesh.faces):
            if face_jdx != face_idx:
                f_j = centroids[face_jdx]
                sigma_s += np.linalg.norm(f_j- f_i)
                num += 1
    return sigma_s * multiple / num"""

import numpy as np

def getSigmaS(multiple, mesh):
    centroids = np.asarray(mdb.getFaceCentroid(mesh))  # shape: (n_faces, 3)

    # Calcola tutte le differenze tra coppie di centroidi
    diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    # Considera solo la parte superiore della matrice (escludi i==j e duplicati)
    triu_indices = np.triu_indices_from(dists, k=1)
    sigma_s = np.sum(dists[triu_indices])
    num = len(triu_indices[0])

    return sigma_s * multiple / num if num > 0 else 0.0


def updateFilteredNormalsWithPredictedNormal(mesh, guided_normals,      normal_iterations_number, dumping_factor):
    # initializating parameters
    denoise_index = 0
    face_neighbor_index = 1
    include_central_face = True
    multiple_radius = 2
    multiple_sigma_s = 1
    normal_iteration_number = normal_iterations_number
    sigma_r = 0.3
    vertex_iteration_number = 16

    filtered_normals = np.zeros_like(guided_normals, dtype=float)

    if face_neighbor_index == 0:
        face_neighbor_type = FaceNeighborType.RADIUS_BASED
    else:
        face_neighbor_type = FaceNeighborType.VERTEX_BASED
    radius = 1.0
    if face_neighbor_type == FaceNeighborType.RADIUS_BASED:
        print("Getting radius", flush=True)
        radius = getRadius(multiple_radius, mesh)
    all_face_neighbor = getAllFaceNeighborGMNF(mesh,
                                                face_neighbor_type,
                                                radius,
                                                include_central_face)
    print(f"All face neighbors shape {len(all_face_neighbor)}", flush=True)
    filtered_normals = mdb.getFaceNormals(mesh)
    for iter in range(normal_iteration_number):
        if iter == 0:
            print(f"Normal filtering...", flush=True)
        else: 
            print(f"Normal Iteration {iter}/{normal_iteration_number}", flush=True)
        face_centroid = mdb.getFaceCentroid(mesh)
        print("Face centroids obtained", flush=True)
        sigma_s = getSigmaS(multiple_sigma_s, mesh)
        print("Sigma s obtained", flush=True)
        face_area = mdb.getFaceArea(mesh)
        print("Face areas obtained", flush=True)
        previous_normals = mdb.getFaceNormals(mesh)
        print("Face normals obtained", flush=True)

        for face_idx, face in enumerate(mesh.faces):
            face_neighbor = all_face_neighbor[face_idx]
            filtered_normal = np.zeros((3,), dtype=float)
            for face_jdx in face_neighbor:
                spatial_distance = np.linalg.norm(face_centroid[face_idx] - face_centroid[face_jdx])
                spatial_weight = gaussianWeight(spatial_distance, sigma_s)
                range_distance = (guided_normals[face_idx] - guided_normals[face_jdx])
                range_weight = gaussianWeight(range_distance, sigma_r)

                if iter == 0:
                    filtered_normal += guided_normals[face_jdx] * (face_area[face_jdx] * spatial_weight * range_weight)
                else: 
                    filtered_normal += previous_normals[face_jdx] * (face_area[face_jdx] * spatial_weight * range_weight)
            if len(face_neighbor):
                norm = np.linalg.norm(filtered_normal)
                if norm > 1e-8:
                    filtered_normals[face_idx] = filtered_normal / np.linalg.norm(filtered_normal)
        new_vertices = mdb.updateVertexPosition(mesh, filtered_normals, iter, fixed_boundary=True, dumping_factor=dumping_factor)
        mesh.vertices = new_vertices
    return mesh


if __name__ == "__main__":
    # Mesh semplice con 2 facce adiacenti
    vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]])
    faces = np.array([[0,1,2], [1,3,2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    print(f"Mesh di test creata con {len(mesh.edges)} facce.")