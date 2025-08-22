import numpy as np
import trimesh

def compute_normal_error(mesh_gt_path, mesh_denoised_path):
    """
    Calcola l'errore angolare medio tra le normali di due mesh (ground truth e denoised).
    
    Parameters:
        mesh_gt_path (str): percorso della mesh ground truth (.obj, .ply, ecc.)
        mesh_denoised_path (str): percorso della mesh denoised
    
    Returns:
        float: errore medio angolare in radianti
        float: errore medio angolare in gradi
    """
    # Carica le mesh
    mesh_gt = trimesh.load(mesh_gt_path, process=True)
    mesh_denoised = trimesh.load(mesh_denoised_path, process=True)

    # Calcola le normali delle facce (già normalizzate in trimesh)
    normals_gt = mesh_gt.face_normals
    normals_denoised = mesh_denoised.face_normals

    # Controllo che abbiano lo stesso numero di facce
    if normals_gt.shape != normals_denoised.shape:
        raise ValueError("Le due mesh non hanno lo stesso numero di facce!")

    # Calcolo del prodotto scalare tra normali
    dot_products = np.einsum('ij,ij->i', normals_gt, normals_denoised)

    # Clipping per evitare errori numerici fuori [-1,1]
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Calcolo degli angoli
    angles = np.arccos(dot_products)

    # Errore medio
    Ea_rad = np.mean(angles)
    Ea_deg = np.degrees(Ea_rad)

    return Ea_rad, Ea_deg


# ESEMPIO USO:
if __name__ == "__main__":
    Ea_rad, Ea_deg = compute_normal_error("ground_truth.obj", "denoised.obj")
    print(f"Errore medio angolare: {Ea_rad:.6f} rad ({Ea_deg:.6f} gradi)")
