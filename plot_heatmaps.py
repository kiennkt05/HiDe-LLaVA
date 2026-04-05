import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_projector(task_dir):
    path = os.path.join(task_dir, "non_lora_trainables.bin")
    if not os.path.exists(path):
        print(f"Warning: Projector path not found: {path}")
        return None
    sd = torch.load(path, map_location="cpu")
    weights = []
    for k in sorted(sd.keys()):
        if "mm_projector" in k:
            weights.append(sd[k].to(torch.float32).flatten())
    if len(weights) == 0:
        return None
    return torch.cat(weights).numpy()

_state_dict_cache = {}

def load_A_matrix(task_dir, target_layer, expert_idx=0):
    path = os.path.join(task_dir, "adapter_model.bin")
    if not os.path.exists(path):
        if target_layer == 0:  # Only warn once per task
            print(f"Warning: Adapter path not found: {path}")
        return None
        
    if path not in _state_dict_cache:
        _state_dict_cache[path] = torch.load(path, map_location="cpu")
    sd = _state_dict_cache[path]
    
    weights = []
    # Ensure stable sorting for concatenation
    for k in sorted(sd.keys()):
        m = re.search(r"layers\.(\d+)\.", k)
        if m and int(m.group(1)) == target_layer:
            if "lora_A" in k:
                m_expert = re.search(r"loraA\.(\d+)", k)
                if m_expert and int(m_expert.group(1)) == expert_idx:
                    weights.append(sd[k].to(torch.float32).flatten())
    if len(weights) == 0:
        return None
    return torch.cat(weights).numpy()

def calc_sim_matrix(vectors):
    N = len(vectors)
    sim_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if vectors[i] is None or vectors[j] is None:
                sim_mat[i, j] = np.nan
            else:
                v1 = torch.tensor(vectors[i])
                v2 = torch.tensor(vectors[j])
                sim_mat[i, j] = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    return sim_mat

def plot_heatmap(sim_mat, title, out_path, task_names):
    plt.figure(figsize=(7, 6))
    
    # If matrix is all NaNs, skip drawing to avoid seaborn errors
    if np.isnan(sim_mat).all():
        print(f"Skipping {out_path} because all data is missing.")
        plt.close()
        return

    ax = sns.heatmap(sim_mat, annot=True, cmap="coolwarm", fmt=".3f", 
                     xticklabels=task_names, yticklabels=task_names,
                     vmin=np.nanmin(sim_mat), vmax=np.nanmax(sim_mat))
    
    # Add borders to cells for better readability
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def get_dir(base_dir, task_id, prefix):
    if task_id == 1:
        return os.path.join(base_dir, "Task1_llava_lora_ours")
    else:
        return os.path.join(base_dir, f"{prefix}_Task{task_id}_llava_lora_ours") if prefix else os.path.join(base_dir, f"Task{task_id}_llava_lora_ours")

def main(base_dir):
    print(f"Using base directory: {base_dir}")
    
    task_names = [f"Task {i}" for i in range(1, 7)]
    
    # 1. hide/projectors.png
    print("\n--- Processing HiDe Projectors ---")
    hide_proj_vecs = [load_projector(get_dir(base_dir, i, "")) for i in range(1, 7)]
    hide_proj_sim = calc_sim_matrix(hide_proj_vecs)
    plot_heatmap(hide_proj_sim, "Cosine Similarity of Projectors (HiDe)", "HeatMap/hide/projectors.png", task_names)
    
    # 2. a/projectors.png & a/a_layer*.png (0-30)
    print("\n--- Processing a_Task Projectors and A Matrices ---")
    a_dirs = [get_dir(base_dir, i, "a") for i in range(1, 7)]
    a_proj_vecs = [load_projector(d) for d in a_dirs]
    a_proj_sim = calc_sim_matrix(a_proj_vecs)
    plot_heatmap(a_proj_sim, "Cosine Similarity of Projectors (Variant A)", "HeatMap/a/projectors.png", task_names)
    
    for l in range(31): # 0 to 30
        vecs = [load_A_matrix(d, l, expert_idx=0) for d in a_dirs]
        sim = calc_sim_matrix(vecs)
        plot_heatmap(sim, f"Cosine Sim of A Matrices (Variant A, Layer {l})", f"HeatMap/a/a_layer{l:02d}.png", task_names)
        
    # 3. fa/projectors.png & fa/fa_layer*.png (0-31)
    print("\n--- Processing fa_Task Projectors and A Matrices ---")
    fa_dirs = [get_dir(base_dir, i, "fa") for i in range(1, 7)]
    fa_proj_vecs = [load_projector(d) for d in fa_dirs]
    fa_proj_sim = calc_sim_matrix(fa_proj_vecs)
    plot_heatmap(fa_proj_sim, "Cosine Similarity of Projectors (Variant fA)", "HeatMap/fa/projectors.png", task_names)
    
    for l in range(32): # 0 to 31
        vecs = [load_A_matrix(d, l, expert_idx=0) for d in fa_dirs]
        sim = calc_sim_matrix(vecs)
        plot_heatmap(sim, f"Cosine Sim of A Matrices (Variant fA, Layer {l})", f"HeatMap/fa/fa_layer{l:02d}.png", task_names)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="HiDe_1")
    args = parser.parse_args()
    main(args.base_dir)
