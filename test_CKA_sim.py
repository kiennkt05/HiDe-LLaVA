import argparse
import torch
import gc
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
from collections import defaultdict
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pickle


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_path = os.path.expanduser(os.path.join(self.image_folder, image_file))
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def hook_fn(name, activation_values):
    def hook(module, input, output):
        if output.size(1) >= 2:
            output_mean = torch.mean(output, dim=1)
            activation_values[name].append(output_mean.detach().cpu())
    return hook

def register_hooks(model, activation_values):
    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.mlp.down_proj.register_forward_hook(hook_fn(f'layer_{i}_final_output', activation_values))
        hooks.append(h)
    return hooks

# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, text_tower=args.text_tower)

    activation_values = defaultdict(list)
    hooks = register_hooks(model, activation_values)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, os.path.expanduser(args.image_folder), tokenizer, image_processor, model.config)

    num = 1
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        num += 1
        if num == 51:
            break
        # ans_file.flush()
    for layer_name, activations in activation_values.items():
        activation_values[layer_name] = torch.cat(activations, dim=0)

    # Memory Cleanup
    for h in hooks:
        h.remove()
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    for layer_name, activations in activation_values.items():
        activation_values[layer_name] = activations.cpu().numpy()
    ans_file.close()

    return activation_values

def compute_l2_drift(X, Y):
    """
    Computes the mean L2 (Euclidean) distance between two sets of activations.
    X and Y should be arrays of shape (num_samples, hidden_dim).
    """
    return np.linalg.norm(X - Y, ord=2, axis=1).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default="~/Documents/kienNguyen/HiDe-LLaVA/llava-7b-v1-5")
    parser.add_argument("--base_dir", type=str, default="~/Documents/kienNguyen/HiDe-LLaVA/HiDe_1")
    parser.add_argument("--run_type", type=str, default="Task")
    parser.add_argument("--image_folder", type=str, default="~/Documents/kienNguyen/HiDe-LLaVA/UCIT/datasets")
    parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--text_tower", type=str, default="/home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/clip-vit-large-patch14-336")
    args = parser.parse_args()

    BASE_MODEL = os.path.expanduser(args.model_base) if args.model_base else None

    question_files = [
        "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/ImageNet-R/test_3000.json",
        "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/ArxivQA/test_3000.json",
        "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/VizWiz/test_3000.json",
        "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/IconQA/test_3000.json",
        "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/CLEVR/test_3000.json",
        "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/Flickr30k/test_3000.json",
    ]
    model_paths = [
        f"{args.base_dir}/Task1_llava_lora_ours",
        f"{args.base_dir}/{args.run_type}2_llava_lora_ours",
        f"{args.base_dir}/{args.run_type}3_llava_lora_ours",
        f"{args.base_dir}/{args.run_type}4_llava_lora_ours",
        f"{args.base_dir}/{args.run_type}5_llava_lora_ours",
        f"{args.base_dir}/{args.run_type}6_llava_lora_ours",
    ]

    os.makedirs(os.path.expanduser(f"./HeatMap/activations/{args.run_type}"), exist_ok=True)
    for idx, model_path in enumerate(model_paths):
        args.model_path = model_path
        args.model_base = BASE_MODEL
        args.text_tower = os.path.expanduser(args.text_tower)
        for i, question_file in enumerate(question_files):
            if i > idx:
                break
            print(f"===> Evaluate {model_path} on {question_file}\n")
            args.question_file = question_file
            act = eval_model(args)
            with open(f'./HeatMap/activations/{args.run_type}/activation_{args.run_type}{idx+1}_task{i+1}.pkl', 'wb') as f:
                pickle.dump(act, f)

    output_dir = os.path.expanduser(f"./HeatMap/CKA/{args.run_type}/")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(6):
        task_idx = i + 1
        available_model_indices = [m_idx + 1 for m_idx in range(len(model_paths)) if m_idx >= i]
        num_models = len(available_model_indices)
        
        if num_models < 2:
            print(f"Skipping CKA for Task {task_idx}: only {num_models} model(s) available.")
            continue

        print(f"Calculating CKA for Task {task_idx} across {num_models} models...")
        
        # Load relevant activations for this dataset
        task_activations = []
        for m_idx in available_model_indices:
            filename = f'./HeatMap/activations/{args.run_type}/activation_{args.run_type}{m_idx}_task{task_idx}.pkl'
            with open(filename, 'rb') as f:
                task_activations.append(pickle.load(f))
        
        labels = [f"{args.run_type}{m_idx}" for m_idx in available_model_indices]

        for layer in range(32):
            kernel_CKA_matrix = np.zeros((num_models, num_models))
            for row in range(num_models):
                for col in range(num_models):
                    if row <= col:
                        res = kernel_CKA(task_activations[row][f'layer_{layer}_final_output'], 
                                        task_activations[col][f'layer_{layer}_final_output'])
                        kernel_CKA_matrix[row, col] = np.round(res, 4)
                        kernel_CKA_matrix[col, row] = kernel_CKA_matrix[row, col]
            
            plt.figure(figsize=(10, 8), dpi=300)
            sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", 
                        vmin=0.0, vmax=1.0, xticklabels=labels, yticklabels=labels)
            plt.title(f"CKA Similarity - Task {task_idx} - Layer {layer}")
            plt.savefig(os.path.join(output_dir, f"CKA_{args.run_type}{task_idx}_layer{layer}.png"))
            plt.close()

    # ==========================================
    # L2 Drift Computation and Visualization
    # ==========================================
    
    l2_output_dir = os.path.expanduser(f"./HeatMap/L2_Drift/{args.run_type}/")
    os.makedirs(l2_output_dir, exist_ok=True)

    # We compute drift for tasks 1 to 5 against all subsequent models
    for task_idx in range(1, 6):
        reference_model_idx = task_idx
        # Find all models that were trained AFTER the reference model
        later_model_indices = [m_idx for m_idx in range(reference_model_idx + 1, 7)]
        
        if not later_model_indices:
            continue
            
        print(f"Calculating L2 Drift for Task {task_idx} (Model {reference_model_idx} vs Models {later_model_indices})...")
        
        # Load the reference activations (the model right after learning the task)
        ref_filename = f'activation_{args.run_type}{reference_model_idx}_task{task_idx}.pkl'
        try:
            with open(ref_filename, 'rb') as f:
                ref_activations = pickle.load(f)
        except FileNotFoundError:
            print(f"  -> Missing reference file: {ref_filename}. Skipping Task {task_idx}.")
            continue
            
        # Dictionary to store drift values for plotting: {model_idx: [drift_layer_0, ..., drift_layer_31]}
        drift_results = {m_idx: [] for m_idx in later_model_indices}
        
        for m_idx in later_model_indices:
            eval_filename = f'activation_{args.run_type}{m_idx}_task{task_idx}.pkl'
            try:
                with open(eval_filename, 'rb') as f:
                    eval_activations = pickle.load(f)
            except FileNotFoundError:
                print(f"  -> Missing evaluation file: {eval_filename}. Skipping comparison.")
                continue
                
            for layer in range(32):
                ref_act = ref_activations[f'layer_{layer}_final_output']
                eval_act = eval_activations[f'layer_{layer}_final_output']
                
                drift = compute_l2_drift(ref_act, eval_act)
                drift_results[m_idx].append(drift)
        
        # Plotting L2 Drift across all layers for the current task
        plt.figure(figsize=(12, 6), dpi=300)
        colors = sns.color_palette("husl", len(later_model_indices))
        
        for i, m_idx in enumerate(later_model_indices):
            if len(drift_results[m_idx]) == 32:
                plt.plot(range(32), drift_results[m_idx], marker='o', markersize=4, 
                         color=colors[i], label=f'Model {m_idx} vs Model {reference_model_idx}')
            
        plt.title(f"L2 Feature Drift across Layers - Task {task_idx} Baseline", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Mean L2 Distance", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(l2_output_dir, f"L2_Drift_{run_type}{task_idx}.png"))
        plt.close()