import numpy as np
import trimesh, os
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def compute_normal_error(mesh_gt_path, mesh_denoised_path):
    mesh_gt = trimesh.load(mesh_gt_path, process=True)
    mesh_denoised = trimesh.load(mesh_denoised_path, process=True)

    normals_gt = mesh_gt.face_normals
    normals_denoised = mesh_denoised.face_normals

    if normals_gt.shape != normals_denoised.shape:
        raise ValueError("Le due mesh non hanno lo stesso numero di facce!")

    dot_products = np.einsum('ij,ij->i', normals_gt, normals_denoised)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products)

    Ea_rad = np.mean(angles)
    Ea_deg = np.degrees(Ea_rad)
    return Ea_deg

def compute_hausdorff_errors(mesh_gt_path, mesh_denoised_path, mesh_denoised_mod_path, n_samples=5000):
    mesh_gt = trimesh.load(mesh_gt_path, process=True)
    mesh_denoised = trimesh.load(mesh_denoised_path, process=True)
    mesh_denoised_mod = trimesh.load(mesh_denoised_mod_path, process=True)

    points_gt, _ = trimesh.sample.sample_surface(mesh_gt, n_samples)
    points_dn, _ = trimesh.sample.sample_surface(mesh_denoised, n_samples)
    points_dn_mod, _ = trimesh.sample.sample_surface(mesh_denoised_mod, n_samples)

    tree_gt = cKDTree(points_gt)

    dist_dn_to_gt, _ = tree_gt.query(points_dn, k=1)
    dist_dn_mod_to_gt, _ = tree_gt.query(points_dn_mod, k=1)

    bbox_min = mesh_gt.bounds[0]
    bbox_max = mesh_gt.bounds[1]
    Ld = np.linalg.norm(bbox_max - bbox_min)

    Nv = n_samples
    Ed_dn = np.sum(dist_dn_to_gt) / (Nv * Ld)
    Ed_dn_mod = np.sum(dist_dn_mod_to_gt) / (Nv * Ld)

    return Ed_dn, Ed_dn_mod

if __name__ == "__main__":
    base_name = "sharp_sphere"
    gt_model_path = os.path.join("testing_models", f"{base_name}_gt.obj")

    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

    Ea_original_list, Ea_modified_list = [], []
    Ed_original_list, Ed_modified_list = [], []

    for noise in noise_levels:
        denoised_path = os.path.join("testing_models", f"denoised_{base_name}_{noise}.obj")
        denoised_mod_path = os.path.join("testing_models", f"denoised_{base_name}_{noise}_mod.obj")

        Ea_rad_original = compute_normal_error(gt_model_path, denoised_path)
        Ea_rad_modified = compute_normal_error(gt_model_path, denoised_mod_path)

        Ea_original_list.append(Ea_rad_original)
        Ea_modified_list.append(Ea_rad_modified)

        Ed_original, Ed_modified = compute_hausdorff_errors(gt_model_path, denoised_path, denoised_mod_path, n_samples=10000)
        Ed_original_list.append(Ed_original)
        Ed_modified_list.append(Ed_modified)

        print(f"Noise {noise}: E_a(orig)={Ea_rad_original:.6f}, E_a(mod)={Ea_rad_modified:.6f}, "
              f"E_d(orig)={Ed_original:.6f}, E_d(mod)={Ed_modified:.6f}")

    # --- Grafico ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # E_a
    axs[0].plot(noise_levels, Ea_original_list, marker="o", label="Originale")
    axs[0].plot(noise_levels, Ea_modified_list, marker="s", label="Modificato")
    for x, y in zip(noise_levels, Ea_original_list):
        axs[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha="center", fontsize=8)
    for x, y in zip(noise_levels, Ea_modified_list):
        axs[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0,5), ha="center", fontsize=8)
    axs[0].set_title("Errore angolare medio $E_a$")
    axs[0].set_xlabel("Livello di rumore \n (a)")
    axs[0].set_ylabel("Errore (gradi)")
    axs[0].legend()
    axs[0].grid(True)

    # E_d
    axs[1].plot(noise_levels, Ed_original_list, marker="o", label="Originale")
    axs[1].plot(noise_levels, Ed_modified_list, marker="s", label="Modificato")
    for x, y in zip(noise_levels, Ed_original_list):
        axs[1].annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,5), ha="center", fontsize=8)
    for x, y in zip(noise_levels, Ed_modified_list):
        axs[1].annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,5), ha="center", fontsize=8)
    axs[1].set_title("Errore Hausdorff normalizzato $E_d$")
    axs[1].set_xlabel("Livello di rumore \n (b)")
    axs[1].set_ylabel("Errore normalizzato")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    out_path = f"{base_name}_error_plot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Grafico salvato come: {out_path}")
