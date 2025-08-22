import numpy as np
import os
import trimesh
import torch
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from GCNModel import DGCNN
    import parsers 
    from datautils import loadMAT
    from mesh_reader_optimized import calc_centroid
except ImportError as e:
    logger.error(f"Error importing dependencies: {e}")
    raise

# --- Helper functions --- 

def find_first_ring_neighborhood(vertex_index, mesh):
    """
    Find indices of faces that shares common vertex.

    Args:
        vertex_index (int): vertex index.
        mesh (trimesh.Trimesh): target mesh.
    Returns:
        Indexes array of faces that shares the vertex.
    """
    if vertex_index >= len(mesh.vertices) or vertex_index < 0:
        return np.array([], dtype=int)
    
    try:
        boolean_matrix = mesh.faces == vertex_index
        faces_first_ring = np.any(boolean_matrix, axis=1)
        return np.where(faces_first_ring)[0]
    except Exception as e:
        logger.warning(f"Error computing first ring for vertex {vertex_index}: {e}")
        return np.array([], dtype=int)

def validate_normals(normals):
    """
    Validate and normalize an array of normals.

    Args:
        normals(np.ndarray): normals array (N,3).

    Returns:
        normals array to normalize.
    """
    if normals.ndim == 1:
        normals = normals.reshape(1,-1)
    
    if normals.shape[1] != 3:
        raise ValueError(
            f"Normals must have 3 components, found {normals.shape[1]}"
        )
    
    norms =np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8]  = 1.0
    return normals / norms

def refine_normals_iteratively(initial_normals, mesh, m=12,
                                 sigma_s_factor=1.0, sigma_r=0.3):
    """
    Apply an iterative bilateral filter to normals field.

    Args:
        initial_normals(np.ndarray): initial normals.
        mesh(ExpandedMesh or trimesh.Trimesh): target mesh. It is suggested to use and ExpandedMesh to optimized elaboration.
        m(int): iterations number.
        sigma_s_factor(float): spatial sigma factor.
        sigma_r(float): range sigma (normals).
    
    Returns:
        np.ndarray: Normals refined
    """
    logger.info(f"Starting normals refinement with bilateral filter for {m} iterations...")

    # Using ExpandedMesh
    if hasattr(mesh, 'centroids') and hasattr(mesh, 'face_adjacency'):
        logger.info('Using ExpandedMesh with pre-calculated properties')
        centroids = mesh.centroids
        face_adjacency = mesh.face_adjacency
        num_faces = len(mesh.faces)

        if hasattr(mesh, 'areas') and len(mesh.areas) > 0:
            avg_edge_len = np.sqrt(np.mean(mesh.areas) * 4.0 / np.sqrt(3))
        else:
            avg_edge_len = 1.0
    # Using Trimesh
    else:
        logger.info('Using trimesh.Trimesh, calculating properties...')
        num_faces = len(mesh.faces)
        if num_faces == 0:
            logger.warning("Mesh without faces, returning original normals")
            return initial_normals

        try:
            centroids = mesh.triangles_center
            if hasattr(mesh, 'edges_unique_length') and len(mesh.avg_edge_len) > 0:
                avg_edge_len = mesh.edges_unique_length.mean()
            else:
                edge_lengths = []
                for face in mesh.faces:
                    for i in range(3):
                        v1, v2 = face[i], face[(i+1)%3]
                        edge_lengths.append(np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2]))
                avg_edge_len = np.mean(edge_lengths) if edge_lengths else 1.0
            
            face_adjacency = mesh.face_adjacency
        except Exception as e:
            logger.warning(
                f"Error during computing of mesh properties: {e}"
            )
            centroids = np.array([np.mean(mesh.vertices[face], axis=0) for face in mesh.faces])
            avg_edge_len = 1.0
            face_adjacency = []
    if num_faces == 0:
        logger.warning("Mesh without faces, returning original normals")
        return initial_normals
    
    sigma_s = avg_edge_len * sigma_s_factor
    sigma_s_sq_2 = 2 * sigma_s**2 + 1e-9
    sigma_r_sq_2 = 2 * sigma_r**2 + 1e-9

    adjacency_list = [[] for _ in range(num_faces)]
    try:
        for face_pair in face_adjacency:
            if 0 <= face_pair[0] < num_faces and 0 <= face_pair[1] < num_faces:
                adjacency_list[face_pair[0]].append(face_pair[1])
                adjacency_list[face_pair[1]].append[face_pair[0]]
    except Exception as e:
        logger.warning(f"Error during computing of adjacency: {e}")

    current_normals = validate_normals(initial_normals.reshape(num_faces, 3))

    for k in range(m):
        normals_k = np.copy(current_normals)
        next_normals = np.zeros_like(normals_k)

        for i in range(num_faces):
            n_i = normals_k[i]
            c_i = centroids[i]
            neighbors_idx = adjacency_list[i]

            all_indices = np.append(neighbors_idx, i)

            if len(all_indices) == 0:
                next_normals[i] = n_i
                continue
            normals_j = normals_k[all_indices]
            centroids_j = centroids[all_indices]

            centroids_diff_sq = np.sum((centroids_j - c_i)**2, axis=1)
            normals_diff_sq = np.sum((normals_j - n_i)**2, axis=1)

            Ws = np.exp(-centroids_diff_sq / sigma_s_sq_2)
            Wr = np.exp(-normals_diff_sq / sigma_r_sq_2)
            weights = WS * Wr

            if np.sum(weights) < 1e-8:
                next_normals[i] = n_i
                continue

            sum_vector = np.sum(weights[:, np.newaxis] * normals_j, axis=0)

            sum_vector_norm = np.linalg.norm(sum_vector)
            if sum_vector_norm > 1e-8:
                next_normals[i] = sum_vector / sum_vector_norm
            else:
                next_normals[i] = n_i 

        current_normals = np.copy(next_normals)
        logger.info(f"Iteration filter {k+1}/{m} completed.")

    logger.info("Normals refinement completed.")
    return current_normals         

def get_boundary_vertices(mesh):
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
        logger.warning(f"Error while computing border vertex: {e}")
        return set()

def predict_normals(mesh_name, dgcnn, k_opt, noise_level, device):
    """
    Predicts normals for all mesh faces.

    Args:
        mesh_name(str): name of the mesh to be processed.
        dgcnn(torch.nn.Module): GCN trained to use to denoise.
        k_opt: parser for arguments.
        noise_level(float): noise level of the mesh.
        device(torch.device): Device to be used for computing.

    Returns:
        np.ndarray: array of predicted normals.
    """
    logger.info("Phase 1: Normal prediction for each face...")

    samples_dir = os.path.join("testing_samples", f"{mesh_name}_{noise_level}")
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    mat_files = [f for f in os.listdir(samples_dir) if f.endswith('.mat')]
    num_faces = len(mat_files)

    if num_faces == 0:
        raise ValueError(f"No .mat file found in {samples_dir}")
    
    logger.info(f"Found {num_faces} .mat files to process")

    predict_normals_list = []

    for i in range(num_faces): 
        mat_path = os.path.join(samples_dir, f"0_{i}.mat")
        if os.path.exists(mat_path):
            try:
                inputs, gt_res, gt_norm, center_norm = loadMAT(mat_path, k_opt.num_neighbors)

                input_features = torch.FloatTensor(inputs).unsqueeze(0).permute(0,2,1).to(device)
                
                with torch.no_grad():
                    output = dgcnn(input_features)
                
                pred_norm = output.cpu().numpy().reshape(3)
                predict_normals_list.append(pred_norm)
            
            except Exception as e:
                logger.warning(
                    f"Error processing patch {i}: {e}. Using default normal."
                )
                predict_normals_list.append(np.array([0.0, 0.0, 1.0]))
        else:
            logger.warning(
                f"Patch file not found: {mat_path}. Using default normal."
            )
            predict_normals_list.append(np.array([0.0, 0.0, 1.0]))
    
    normals_array = np.array(predict_normals_list)
    logger.info("Normals prediction completed.")
    return validate_normals(normals_array)

def update_vertex_positions(mesh, normals, k_iterations=100, lambda_factor=0.5):
    """
    Updates vertex positions based on denoised normals.

    Args:
        mesh(trimesh.Trimesh or ExpandedMesh): original mesh.
        normals(np.ndarray): denoised normals.
        k_iterations(int): number of iterations.
        lambda_factor(float): attenuation factor of vertex displacement.

    Returns:
        np.ndarray: vertex new positions
    """
    logger.info("Updating vertex position.")

    new_vertices = np.copy(mesh.vertices)
    num_vertices = len(mesh.vertices)

    use_precalculated_centroids = hasattr(mesh, 'centroids')
    if use_precalculated_centroids:
        logger.info("Using centroids of ExpandedMesh structure.")   

    boundary_vertices = get_boundary_vertices(mesh)
    logger.info(f"Found {len(boundary_vertices)} border vertex to set.")

    for k in range(k_iterations):
        vertices_k = np.copy(new_vertices)
        vertices_k_plus_1 = np.zeros_like(vertices_k)
        total_update_magnitude = 0.0
        processed_vertices = 0

        for i in range(num_vertices):
            v_i_k = vertices_k[i]

            if i in boundary_vertices:
                vertices_k_plus_1[i] = v_i_k
                continue
            
            first_ring_neighbors = find_first_ring_neighborhood(i, mesh)

            if len(first_ring_neighbors) == 0:
                vertices_k_plus_1[i] = v_i_k

            sum_faces_contribution = np.zeros(3, dtype=float)

            for face_idx in first_ring_neighbors:
                denoised_normal = normals[face_idx].reshape(3)
                if use_precalculated_centroids:
                    face_centroid = mesh.centroids[face_idx]
                else:
                    try:
                        face_centroid = calc_centroid(
                            mesh.faces[face_idx], vertices_k
                        )
                    except Exception:
                        face_vertices = vertices_k[mesh.faces[face_idx]]
                        face_centroid = np.mean(face_vertices, axis=0)
                
                diff = face_centroid - v_i_k
                projection_scalar = np.dot(denoised_normal, diff)
                term_vector = denoised_normal * projection_scalar
                sum_faces_contribution += term_vector
            
            if len(first_ring_neighbors) > 0:
                norm_factor = 1.0/(3.0 * len(first_ring_neighbors))
                update_vector = norm_factor * sum_faces_contribution
                actual_update = lambda_factor * update_vector

                total_update_magnitude += np.linalg.norm(actual_update)
                vertices_k_plus_1[i] = v_i_k + actual_update
                processed_vertices += 1
            else:
                vertices_k_plus_1[i] = v_i_k
        new_vertices = np.copy(vertices_k_plus_1)

        avg_update = total_update_magnitude / processed_vertices 
        logger.info(f"Iteration {k+1}/{k_iterations}, Average magnitude of update: {avg_update:.6f}")

    logger.info("Vertex update completed.")
    return new_vertices

def surface_denoising(mesh_name, noise_level, k_opt, use_refinement = True, use_expanded_mesh=True):
    """
    Execute entire denoising process: normals prediction, refinement and vertex positions update. 

    Args:
        mesh_name(str): Name of the mesh to process.
        noise_level(float): level of noise of selected noised mesh.
        use_refinement(bool): Decide whether to use bilateral filtering or not.
        use_expandend_mesh(bool): Decide whether use ExpandendMesh for optimization 
    """
    try:
        mesh_path = os.path.join(f'testing_models', f"{mesh_name}_noised_{noise_level}_Gaussian.obj")

        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        
        logger.info(f"Mesh loading: {mesh_path}")
        original_mesh = trimesh.load_mesh(mesh_path)

        if len(original_mesh.faces) == 0:
            raise ValueError("Mesh doesn't contain faces")
        
        if use_expanded_mesh:
            logger.info("Creating ExpandedMesh for optimization")
            try:
                from ExpandedMesh import ExpandedMesh
                mesh = ExpandedMesh(original_mesh)
                logger.info("ExpandedMesh created!")
            except ImportError:
                logger.warning("ExpandedMesh not available, using standard trimesh.")
                mesh = original_mesh
        else: 
            mesh = original_mesh

        logger.info(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")

        logger.info(f"Loading model pre-trained: {k_opt.current_model}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(k_opt.current_model):
            raise FileNotFoundError(f"Model file not found: {k_opt.current_model}")
        
        dgcnn = DGCNN(8, 16, 1024, 0.5)
        dgcnn.load_state_dict(torch.load(k_opt.current_model, map_location=device))
        dgcnn.to(device)
        dgcnn.eval()
        logger.info(f"Model successfully updated on {device}")

        predicted_normals = predict_normals(mesh_name, dgcnn, k_opt, noise_level, device)

        if use_refinement:
            logger.info("Applying bilateral filter...")
            normals_to_use = refine_normals_iteratively(predicted_normals, mesh, m=12)
        else:
            normals_to_use = predicted_normals
            logger.info('Bilateral filter deactivated')
        
        new_vertices = update_vertex_positions(mesh, normals_to_use)

        denoised_mesh = trimesh.Trimesh(vertices=new_vertices,
        faces=original_mesh.faces, process=False)

        output_dir = "testing_models"
        os.makedirs(output_dir, exist_ok=True)
        
        denoised_mesh_path = os.path.join(output_dir,
                             f"denoised_{mesh_name}_{noise_level}.obj")
        denoised_mesh.export(file_obj=denoised_mesh_path)
        logger.info(f"Denoised mesh saved in: {denoised_mesh_path}")

        return denoised_mesh_path
    except Exception as e:
        logger.error(f"Error during denoising: {e}")
        raise

def main():
    """
    Main function with arguments handling.
    """
    parser = argparse.ArgumentParser(
        description='Denoise a 3D mesh using a pre-trained GCN-model.'
    )
    parser.add_argument('mesh_name', type=str,
                        help='Name of the mesh to process')
    parser.add_argument('noise_level', type=float,
                        help='Noise level of selected mesh')
    parser.add_argument('--use-refinement', action='store_true',
                        help='Apply bilateral filter to refine normals') 
    parser.add_argument('--disable-expanded-mesh', action='store_true',
                        help='Disable ExpandedMesh optimization')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set logging level.') 

    args = parser.parse_args() 

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        k_opt = parsers.getParser()

        if not hasattr(k_opt, 'current_model') or not k_opt.current_model:
            raise ValueError("Model path not specified in k_opt.current_model")

        output_path = surface_denoising(
            args.mesh_name,
            args.noise_level,
            k_opt,
            args.use_refinement,
            not args.disable_expanded_mesh)
        logger.info(f"Denoising completed. Output: {output_path}")    
    except Exception as e:
        logger.error(f"Error during excecution: {e}")
        return 1

    return 0

if __name__=="__main__":
    exit(main())          