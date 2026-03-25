import torch
import os
import glob

def load_lora_weights(path):
    files = glob.glob(os.path.join(path, "adapter_model.safetensors")) + \
            glob.glob(os.path.join(path, "checkpoint-*/adapter_model.safetensors"))
    if files:
        from safetensors.torch import load_file
        return load_file(files[-1]) # use latest
        
    files = glob.glob(os.path.join(path, "adapter_model.bin")) + \
            glob.glob(os.path.join(path, "checkpoint-*/adapter_model.bin"))
    if files:
        return torch.load(files[-1], map_location="cpu", weights_only=True)
        
    raise FileNotFoundError(f"No adapter_model weights found near {path}")

def load_projector_weights(path):
    files = glob.glob(os.path.join(path, "non_lora_trainables.bin")) + \
            glob.glob(os.path.join(path, "checkpoint-*/non_lora_trainables.bin"))
    if files:
        return torch.load(files[-1], map_location="cpu", weights_only=True)
    return None

def print_drift_stats(name, keys, sd1, sd2):
    if not keys:
        print(f"\nWarning: Could not find any keys for {name} in Task 1 weights.")
        return
        
    print(f"\nComparing {len(keys)} tensors for {name}...")
    identical = True
    t1_list, t2_list = [], []
    
    for k in keys:
        if k not in sd2:
            print(f"  ❌ Missing key in Task 2: {k}")
            identical = False
            continue
            
        t1, t2 = sd1[k].float(), sd2[k].float()
        t1_list.append(t1.flatten())
        t2_list.append(t2.flatten())
        
        if not torch.allclose(t1, t2, atol=1e-6):
            identical = False
            
    if identical:
        print(f"✅ {name} weights are 100% IDENTICAL!")
    else:
        if t1_list and t2_list:
            t1_all = torch.cat(t1_list)
            t2_all = torch.cat(t2_list)
            
            max_diff = (t1_all - t2_all).abs().max().item()
            mae = (t1_all - t2_all).abs().mean().item()
            l2_dist = torch.norm(t1_all - t2_all).item()
            avg_magnitude = t1_all.abs().mean().item()
            
            if avg_magnitude > 1e-9 and t2_all.abs().mean().item() > 1e-9:
                cos_sim = torch.nn.functional.cosine_similarity(t1_all.unsqueeze(0), t2_all.unsqueeze(0)).item()
            else:
                cos_sim = float('nan')
                
            print(f"❌ {name} weights CHANGED!")
            print(f"  -> Max Absolute Shift:  {max_diff:.6f}")
            print(f"  -> Mean Absolute Shift: {mae:.6f}  (Average weight magnitude is {avg_magnitude:.6f})")
            print(f"  -> Euclidean (L2) Dist: {l2_dist:.6f}")
            print(f"  -> Cosine Similarity:   {cos_sim:.6f}")

def compare_state_dicts(path1, path2):
    print(f"Loading weights from:\n  Task 1: {path1}\n  Task 2: {path2}\n")
    
    try:
        sd1 = load_lora_weights(path1)
        sd2 = load_lora_weights(path2)
        
        loraA_keys = [k for k in sd1.keys() if '.loraA.0.' in k]
        loraB_keys = [k for k in sd1.keys() if '.loraB.0.' in k]
        
        print_drift_stats("Expert 0 loraA (loraA[0])", loraA_keys, sd1, sd2)
        print_drift_stats("Expert 0 loraB (loraB[0])", loraB_keys, sd1, sd2)
                
    except Exception as e:
        print(f"Could not load LoRA weights: {e}")

    try:
        p1 = load_projector_weights(path1)
        p2 = load_projector_weights(path2)
        
        if p1 and p2:
            proj_keys = [k for k in p1.keys() if 'mm_projector' in k]
            print_drift_stats("mm_projector", proj_keys, p1, p2)
        elif p1 is None or p2 is None:
            print("\nWarning: Could not find non_lora_trainables.bin in both directories to compare projector weights.")
    except Exception as e:
        print(f"Could not load projector weights: {e}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--prev_dir", type=str, required=True)
    parser.add_argument("--cur_dir", type=str, required=True)
    args = parser.parse_args()
    compare_state_dicts(args.prev_dir, args.cur_dir)
