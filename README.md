# Mesh Denoising with Graph Convolutional Networks — Thesis Project

A codebase and experiments for a master's thesis on applying Graph Convolutional Networks (DGCNN-style) to mesh denoising. The implementation focuses on predicting face/vertex normals from local mesh patches and using the predicted normals to drive surface denoising/vertex updates.

Repository: https://github.com/fronzofronzo/mesh-denoising-tesi

Note: I inspected the repository to tailor this README. The file listing used to generate this README may be incomplete; view the full repository in the link above.

## Summary
This project implements:
- a DGCNN-like GCN (GCNModel.py / DGCNN) that predicts normals from local patch features,
- preprocessing and patch/neighbor generation utilities (patch_generator_mod.py, main_patch_generator.py, patch_generator_mod_optimized.py),
- training and experiment orchestration (train.py, optuna_train.py),
- a pipeline to denoise meshes using predicted normals (mesh_denoising_main.py + mesh_normal_filtering.py),
- tools to generate and add noise to meshes (noise_generator.py, generate_noise.py),
- evaluation utilities (normal_error.py, check_new_features.py, datautils.py).

Key directories:
- checkpoints/ — saved checkpoints (empty in repo, used at runtime)
- models/ — for saving/exporting trained models
- testing_models/ — meshes used in testing and outputs
- validationRes/ — validation results
- logs/ — training/experiment logs

Several pre-generated result images and diagnostic plots are included (error plots, diagnostic_report_*.png).

## Quick start

1. Prerequisites
   - Python 3.8+ recommended
   - PyTorch (must match your CUDA / CPU setup)
   - numpy, h5py, trimesh, matplotlib
   - tensorboardX (used in train.py)
   - optuna (optional, for optuna_train.py)
   - any additional requirements used by datautils/patch generator (scipy, sklearn, etc.)
   Install typical packages:
   pip install torch numpy h5py trimesh matplotlib tensorboardX optuna

2. Prepare data
   - Use the patch generator scripts to create training samples / .mat files:
     - main_patch_generator.py or patch_generator_mod.py / patch_generator_mod_optimized.py
   - train.py expects an HDF5 file containing a dataset path list (see datautils.MatrixDataset and getParser in parsers.py). Create/point to your dataset and adjust parser options.

3. Train a model
   - Default training entrypoint:
     python train.py
   - Training behavior and paths are configured via the parser returned by parsers.getParser(). Common config items:
     - k_opt.data_path_file — path to HDF5 file listing sample paths
     - k_opt.ckpt_path — directory to save model checkpoints
     - k_opt.val_res_path — validation result path
     - k_opt.batch_size, k_opt.num_epoch, k_opt.learning_rate, k_opt.num_neighbors, etc.
   - Logs are written under logs/ (the script uses logs/modified_add_layer by default) and to TensorBoard runs/

4. Denoise a mesh using a trained model
   - Use mesh_denoising_main.py which:
     - loads a trained model (configured in k_opt.current_model),
     - expects prepared patch files for the target mesh in new_testing_samples/<mesh>_<noise_level>/ (0_i.mat style),
     - predicts normals for faces and applies the mesh_normal_filtering pipeline to produce a denoised mesh.
   - Example:
     python mesh_denoising_main.py <mesh_name> <noise_level> <normal_iterations_number> <dumping_factor> --use-refinement
   - Outputs are saved in testing_models/ as denoised_<mesh>_<noise_level>_mod.obj

5. Noise generation & testing
   - noise_generator.py and generate_noise.py provide utilities to add synthetic noise to meshes used for training/testing.
   - normal_error.py can be used to compute normal prediction/denoising errors.

6. SLURM / cluster jobs
   - Several .sbatch job scripts are included (job_train.sbatch, job_patch_rtx.sbatch, job_noise_generator.sbatch, job_optuna.sbatch, etc.) to run training, patch generation, noise generation and hyperparameter searches on a cluster. Adapt them to your cluster environment.

## Main files explained
- GCNModel.py — DGCNN implementation (DGCNN) and helper functions (knn, get_graph_feature_idx, get_graph_feature).
- train.py — training loop, validation, logging, checkpoint saving; uses MatrixDataset from datautils and DGCNN.
- mesh_denoising_main.py — high-level denoising pipeline that loads a trained model, predicts normals for a noisy mesh, and applies normal-based surface denoising.
- mesh_normal_filtering.py — routines to update mesh vertices based on filtered/predicted normals (classical filters + integration with predicted normals).
- datautils.py — dataset helpers, MatrixDataset class and loaders for .mat or HDF5 sample lists.
- patch_generator_mod.py / patch_generator_mod_optimized.py — scripts to generate local patch files used as network input (heavy/optimized implementations included).
- main_patch_generator.py — driver to produce testing patches (new_testing_samples/).
- noise_generator.py, generate_noise.py — add synthetic noise to meshes for training/validation.
- optuna_train.py — optional script to run hyperparameter tuning with Optuna.
- check_new_features.py — diagnostics for feature distributions / patch quality.
- parsers.py — central argument/config parser used by scripts (train, denoising, patch generation). Inspect it to see required config options and default paths.

## Typical CLI examples
- Train (default behavior relies on parsers.getParser to supply configuration):
  python train.py
- Run denoising on a mesh using a trained model (k_opt.current_model must point to a .t7/.pth checkpoint in config):
  python mesh_denoising_main.py Bunny 0.02 10 0.5 --use-refinement
- Generate patches for a mesh (example):
  python main_patch_generator.py --mesh Bunny --noise-level 0.02 --out-dir new_testing_samples/Bunny_0.02
- Add noise to a mesh:
  python noise_generator.py --input testing_models/Bunny.obj --sigma 0.02 --output testing_models/Bunny_noised_0.02.obj

Read parsers.py to see exact argument names and defaults.

## Evaluation & results
- The repo includes plotting/diagnostic images (e.g., sharp_sphere_error_plot.png, trim-star_error_plot.png, chinese_lion_error_plot.png) to illustrate per-model error curves and qualitative comparisons.
- normal_error.py and diagnostics in check_new_features.py assist with quantitative evaluation (L2 error, cosine deviation, heatmaps).

## Reproducibility tips
- Save random seeds, full parser config and HDF5 sample lists (train.py records training logs and saves json/csv logs).
- Keep model checkpoints in ckpt_path and validation indices saved in val_res_path (train.py uses val_index.npy).
- Use the provided SLURM job scripts as templates for cluster runs and hyperparameter sweeps.

## Who should use this repo
- Researchers or students working on mesh processing, normal estimation, or graph neural networks for geometry.
- Anyone looking to reproduce the thesis experiments or adapt the DGCNN-based pipeline to other mesh denoising tasks.

## Contributing
- Open an issue in the repository for bug reports or feature requests.
- If you want help reproducing experiments or need datasets used in the thesis, open an issue or contact the repository owner.

## License & citation
- Add your preferred license file (e.g., MIT) to the repository if you want reuse.
- Suggested citation:
  [Author], "Study and application of GCN for mesh denoising," Master's thesis, [Institution], [Year].

Contact / repository owner: https://github.com/fronzofronzo

GitHub Copilot Chat Assistant