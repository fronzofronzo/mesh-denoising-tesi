import trimesh
import os

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "statue.obj")
    mesh = trimesh.load_mesh(model_path)
    print(mesh.vertices[mesh.faces])
    print(mesh.vertices[mesh.faces].shape)