import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import re

def plot_tsne(features, labels_task, labels_type, title, filename_suffix, labels_layer=None):
    if len(features) < 2:
        print(f"Not enough features for {title}, skipping...")
        return
        
    features = np.array(features)
    print(f"[{title}] Feature shape: {features.shape}")
    
    # Pre-dimensional reduction with PCA to save time
    if features.shape[1] > 50:
        pca_dim = min(50, len(features) - 1)
        if pca_dim > 0:
            print(f"[{title}] Running PCA to reduce dimensions to {pca_dim}...")
            pca = PCA(n_components=pca_dim, random_state=42)
            features = pca.fit_transform(features)
    
    perplexity = min(30, max(1, len(features) // 3))
    print(f"[{title}] Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    unique_tasks = list(dict.fromkeys(labels_task)) 
    unique_types = list(dict.fromkeys(labels_type))
    
    if len(unique_types) > 1:
        color_map = plt.get_cmap('tab20')
        get_c = lambda t_idx, m_idx: color_map((t_idx * len(unique_types) + m_idx) % 20)
    else:
        color_map = plt.get_cmap('tab10')
        get_c = lambda t_idx, m_idx: color_map(t_idx % 10)
        
    marker_list = ['o', 'x', 's', '^', 'D', 'p', '*', 'h']
    
    for idx_t, task_name in enumerate(unique_tasks):
        for idx_m, m_type in enumerate(unique_types):
            indices = [i for i, (t, type_) in enumerate(zip(labels_task, labels_type)) if t == task_name and type_ == m_type]
            if indices:
                plt.scatter(
                    reduced[indices, 0], 
                    reduced[indices, 1], 
                    label=f"{task_name} ({m_type})",
                    color=get_c(idx_t, idx_m),
                    marker=marker_list[idx_m % len(marker_list)],
                    alpha=0.7,
                    s=60
                )
                
                if labels_layer is not None:
                    for i in indices:
                        if labels_layer[i] is not None and labels_layer[i] != "":
                            plt.annotate(str(labels_layer[i]), 
                                         (reduced[i, 0], reduced[i, 1]), 
                                         xytext=(0, 5),
                                         textcoords='offset points',
                                         ha='center',
                                         fontsize=7,
                                         alpha=0.85)
                    
    plt.title(title)
    # Move legend out of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, linestyle='--')
    out_path = f"tSNE/{filename_suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}\n")
    plt.close()

def main():
    base_dir = "HiDe_1"
    
    task_names = {
        1: "Image-R",
        2: "ArxivQA",
        3: "Viz-cap",
        4: "IconQA",
        5: "CLEVR",
        6: "Flickr30k"
    }

    matrices_A = []
    labels_A_task = []
    labels_A_type = []
    labels_A_layer = []
    
    matrices_B = []
    labels_B_task = []
    labels_B_type = []
    labels_B_layer = []

    projectors = []
    labels_proj_task = []
    labels_proj_type = []
    labels_proj_layer = []

    anchors_text = []
    labels_txt_anchor_task = []
    labels_txt_anchor_type = []
    labels_txt_anchor_layer = []

    anchors_img = []
    labels_img_anchor_task = []
    labels_img_anchor_type = []
    labels_img_anchor_layer = []
    
    # Process each task and type
    for task_id in range(1, 7):
        for run_type in ["fa_Task", "a_Task", "Task"]:
            if task_id == 1 and run_type != "Task":
                continue
            task_dir = os.path.join(base_dir, f"{run_type}{task_id}_llava_lora_ours")
            adapter_path = os.path.join(task_dir, "adapter_model.bin")
            non_lora_path = os.path.join(task_dir, "non_lora_trainables.bin")
            
            cur_task = task_id - 1
            
            if os.path.exists(adapter_path):
                state_dict = torch.load(adapter_path, map_location="cpu")
                
                layer_A_weights = {l: [] for l in range(32)}
                layer_B_weights = {l: [] for l in range(32)}
                
                for k, v in state_dict.items():
                    m = re.search(r"layers\.(\d+)\.", k)
                    if m:
                        layer_idx = int(m.group(1))
                        if "lora_A" in k:
                            m_expert = re.search(r"loraA\.(\d+)", k)
                            if m_expert:
                                expert_idx = int(m_expert.group(1))
                                
                                is_match = False
                                if run_type == "a_Task" and ((expert_idx == 0 and layer_idx < 31) or (expert_idx == cur_task and layer_idx == 31)):
                                    is_match = True
                                elif run_type == "fa_Task" and expert_idx == 0:
                                    is_match = True
                                elif run_type == "Task" and expert_idx == cur_task:
                                    is_match = True
                                
                                if is_match:
                                    layer_A_weights[layer_idx].append(v.to(torch.float32).flatten().numpy())
                                    
                        elif "lora_B" in k:
                            m_expert = re.search(r"loraB\.(\d+)", k)
                            if m_expert:
                                expert_idx = int(m_expert.group(1))
                                if expert_idx == cur_task:
                                    layer_B_weights[layer_idx].append(v.to(torch.float32).flatten().numpy())

                for l in range(32):
                    if len(layer_A_weights[l]) > 0:
                        flattened_A = np.concatenate(layer_A_weights[l])
                        matrices_A.append(flattened_A)
                        labels_A_task.append(task_names.get(task_id, f"Task{task_id}"))
                        labels_A_type.append(run_type)
                        labels_A_layer.append(l)
                    
                    if len(layer_B_weights[l]) > 0:
                        flattened_B = np.concatenate(layer_B_weights[l])
                        matrices_B.append(flattened_B)
                        labels_B_task.append(task_names.get(task_id, f"Task{task_id}"))
                        labels_B_type.append(run_type)
                        labels_B_layer.append(l)
            else:
                print(f"Warning: {adapter_path} not found. Ensure script runs where HiDe_1 is located.")

            # Projectors and Anchors
            if os.path.exists(non_lora_path):
                nl_state_dict = torch.load(non_lora_path, map_location="cpu")
                
                proj_weights = []
                for k, v in nl_state_dict.items():
                    if "mm_projector" in k:
                        proj_weights.append(v.to(torch.float32).flatten().numpy())
                
                if len(proj_weights) > 0:
                    flattened_proj = np.concatenate(proj_weights)
                    projectors.append(flattened_proj)
                    labels_proj_task.append(task_names.get(task_id, f"Task{task_id}"))
                    labels_proj_type.append(run_type)
                    labels_proj_layer.append(None)
                
                if task_id < 6:
                    continue

                # Only counting the first 6 text anchors and 6 image anchors
                for k, v in nl_state_dict.items():
                    key_tail = k.split(".")[-1]
                    if "text_anchor" in k and key_tail.isdigit() and int(key_tail) < 6:
                        anchors_text.append(v.to(torch.float32).flatten().numpy())
                        labels_txt_anchor_task.append(task_names.get(task_id, f"Task{task_id}"))
                        labels_txt_anchor_type.append(run_type)
                        labels_txt_anchor_layer.append(f"T{key_tail}")
                    elif "image_anchor" in k and key_tail.isdigit() and int(key_tail) < 6:
                        anchors_img.append(v.to(torch.float32).flatten().numpy())
                        labels_img_anchor_task.append(task_names.get(task_id, f"Task{task_id}"))
                        labels_img_anchor_type.append(run_type)
                        labels_img_anchor_layer.append(f"I{key_tail}")
            else:
                pass # Already printed warning for adapter_path, avoiding repeated warnings setup

    os.makedirs("tSNE", exist_ok=True)
    
    def plot_three_variants(features, labels_task, labels_type, labels_layer, base_title, base_filename):
        # 1. Combined
        plot_tsne(features, labels_task, labels_type, base_title, base_filename, labels_layer=labels_layer)
        
        # 2. _hide (only 'Task')
        hide_indices = [i for i, t in enumerate(labels_type) if t == 'Task' or t.startswith('Task ')]
        if hide_indices:
            hide_features = [features[i] for i in hide_indices]
            hide_labels_task = [labels_task[i] for i in hide_indices]
            hide_labels_type = [labels_type[i] for i in hide_indices]
            hide_labels_layer = [labels_layer[i] for i in hide_indices]
            plot_tsne(hide_features, hide_labels_task, hide_labels_type, base_title + " (HiDe)", base_filename + "_hide", labels_layer=hide_labels_layer)
            
        # 3. _a (only 'a_Task')
        a_indices = [i for i, t in enumerate(labels_type) if t == 'a_Task' or t.startswith('a_Task ')]
        if a_indices:
            a_features = [features[i] for i in a_indices]
            a_labels_task = [labels_task[i] for i in a_indices]
            a_labels_type = [labels_type[i] for i in a_indices]
            a_labels_layer = [labels_layer[i] for i in a_indices]
            plot_tsne(a_features, a_labels_task, a_labels_type, base_title + " (Variant A)", base_filename + "_a", labels_layer=a_labels_layer)

        # 4. _fa (only 'fa_Task')
        fa_indices = [i for i, t in enumerate(labels_type) if t == 'fa_Task' or t.startswith('fa_Task ')]
        if fa_indices:
            fa_features = [features[i] for i in fa_indices]
            fa_labels_task = [labels_task[i] for i in fa_indices]
            fa_labels_type = [labels_type[i] for i in fa_indices]
            fa_labels_layer = [labels_layer[i] for i in fa_indices]
            plot_tsne(fa_features, fa_labels_task, fa_labels_type, base_title + " (Variant fA)", base_filename + "_fa", labels_layer=fa_labels_layer)

    # 1. Plot Matrix A
    print(f"Collected {len(matrices_A)} matrices for A. Expected: 384")
    plot_three_variants(matrices_A, labels_A_task, labels_A_type, labels_A_layer, "t-SNE Visualization of Matrices A", "A")
    
    # 2. Plot Matrix B
    print(f"Collected {len(matrices_B)} matrices for B. Expected: 384")
    plot_three_variants(matrices_B, labels_B_task, labels_B_type, labels_B_layer, "t-SNE Visualization of Matrices B", "B")
    
    # 3. Plot Projectors
    print(f"Collected {len(projectors)} projectors. Expected: 12")
    plot_three_variants(projectors, labels_proj_task, labels_proj_type, labels_proj_layer, "t-SNE Visualization of Projectors", "projectors")
    
    # 4. Plot Anchors
    all_anchors = anchors_text + anchors_img
    all_labels_task = labels_txt_anchor_task + labels_img_anchor_task
    all_labels_type = [f"{t} (Text)" for t in labels_txt_anchor_type] + [f"{t} (Image)" for t in labels_img_anchor_type]
    all_labels_layer = labels_txt_anchor_layer + labels_img_anchor_layer
    
    print(f"Collected {len(all_anchors)} anchors. Expected: 24")
    plot_three_variants(all_anchors, all_labels_task, all_labels_type, all_labels_layer, "t-SNE Visualization of Anchors", "anchors")

if __name__ == "__main__":
    main()
