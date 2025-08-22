import mesh_normal_filtering as mnf
import trimesh, os, argparse, torch
import numpy as np
from GCNModel import DGCNN
import parsers
from datautils import loadMAT

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

def predict_normals(mesh_name, dgcnn, k_opt, noise_level, device):
    """
    Predicts normals for all mesh faces with debug info.
    """
    print("Phase 1: Normal prediction for each face...")

    samples_dir = os.path.join("new_testing_samples", f"{mesh_name}_{noise_level}")
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    mat_files = [f for f in os.listdir(samples_dir) if f.endswith('.mat')]
    num_faces = len(mat_files)

    if num_faces == 0:
        raise ValueError(f"No .mat file found in {samples_dir}")
    
    print(f"Found {num_faces} .mat files to process", flush=True)

    predict_normals_list = []

    for i in range(num_faces): 
        mat_path = os.path.join(samples_dir, f"0_{i}.mat")
        if os.path.exists(mat_path):
            try:
                # Caricamento patch
                inputs, gt_res, gt_norm, center_norm = loadMAT(mat_path, k_opt.num_neighbors)
                print(f"[DEBUG] inputs.shape = {inputs.shape}, inputs dtype = {inputs.dtype}")
                print(f"[DEBUG] First row of inputs = {inputs[0]}")

                # Preparazione features per il modello
                input_features = torch.FloatTensor(inputs).unsqueeze(0).permute(0,2,1).to(device)
                print(f"[DEBUG] input_features.shape (after permute) = {input_features.shape}, device = {device}")

                # Forward pass
                with torch.no_grad():
                    output = dgcnn(input_features)
                print(f"[DEBUG] output.shape = {output.shape}")

                # reshape pred normal
                pred_norm = output.cpu().numpy().reshape(3)
                print(f"[DEBUG] pred_norm shape after reshape = {pred_norm.shape}")
                predict_normals_list.append(pred_norm)
            
            except Exception as e:
                print(f"[ERROR] Processing patch {i}: {e}. Using default normal.", flush=True)
                predict_normals_list.append(np.array([0.0, 0.0, 1.0]))
        else:
            print(f"[WARNING] Patch file not found: {mat_path}. Using default normal.")
            predict_normals_list.append(np.array([0.0, 0.0, 1.0]))
    
    normals_array = np.array(predict_normals_list)
    print("Normals prediction completed.", flush=True)
    return validate_normals(normals_array)

def surface_denoising(mesh_name, noise_level, k_opt, normal_iterations_number, dumping_factor):
    mesh_path = os.path.join(f'testing_models', f"{mesh_name}_noised_{noise_level}_Gaussian.obj")
    original_mesh = trimesh.load_mesh(mesh_path)
    print(f"Mesh {mesh_name} loaded", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dgcnn = DGCNN(8,18,1024,0.5)
    dgcnn.load_state_dict(torch.load(k_opt.current_model, map_location=device))
    dgcnn.to(device)
    dgcnn.eval()
    print(f"Model successfully updated on {device}", flush=True)
    predicted_normals = predict_normals(mesh_name, dgcnn, k_opt, noise_level, device)

    denoised_mesh = mnf.updateFilteredNormalsWithPredictedNormal(original_mesh, predicted_normals, normal_iterations_number, dumping_factor)

    output_dir = "testing_models"
    os.makedirs(output_dir, exist_ok=True)
    
    denoised_mesh_path = os.path.join(output_dir,
                            f"denoised_{mesh_name}_{noise_level}_mod.obj")
    denoised_mesh.export(file_obj=denoised_mesh_path)
    print(f"Denoised mesh saved in: {denoised_mesh_path}")

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
    parser.add_argument('normal_iterations_number', type=int,
                        help='Normal iterations number')
    parser.add_argument('dumping_factor', type=float,
                        help='Dumping factor to normal in vertex updating')
    parser.add_argument('--use-refinement', action='store_true',
                        help='Apply bilateral filter to refine normals') 
    parser.add_argument('--disable-expanded-mesh', action='store_true',
                        help='Disable ExpandedMesh optimization')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set logging level.') 

    args = parser.parse_args() 
    print("Args passed!")

    try:
        k_opt = parsers.getParser()

        if not hasattr(k_opt, 'current_model') or not k_opt.current_model:
            raise ValueError("Model path not specified in k_opt.current_model")

        output_path = surface_denoising(
            args.mesh_name,
            args.noise_level,
            k_opt,
            args.normal_iterations_number,
            args.dumping_factor)
        print(f"Denoising completed. Output: {output_path}")    
    except Exception as e:
        print(f"Error during excecution: {e}")
        return 1

    return 0

if __name__=="__main__":
    print("Script started", flush=True)
    exit(main())     