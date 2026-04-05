import argparse
import torch
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

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
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
    for i, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.register_forward_hook(hook_fn(f'layer_{i}_final_output', activation_values))

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
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    activation_values = defaultdict(list)
    register_hooks(model, activation_values)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

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
    for layer_name, activations in activation_values.items():
        activation_values[layer_name] = activations.cpu().numpy()
    ans_file.close()

    return activation_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default="~/Documents/kienNguyen/HiDe-LLaVA/llava-7b-v1-5")
    parser.add_argument("--base_dir", type=str, default="~/Documents/kienNguyen/HiDe-LLaVA/HiDe_1")
    parser.add_argument("--run_type", type=str, default="Task")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    BASE_MODEL = os.path.expanduser(args.model_base) if args.model_base else None

    configs = [
        {"question_file": "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/ImageNet-R/test_3000.json",
         "model_path": f"{args.base_dir}/Task1_llava_lora_ours"},
        {"question_file": "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/ArxivQA/test_3000.json",
         "model_path": f"{args.base_dir}/{args.run_type}2_llava_lora_ours"},
        {"question_file": "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/VizWiz/test_3000.json",
         "model_path": f"{args.base_dir}/{args.run_type}3_llava_lora_ours"},
        {"question_file": "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/IconQA/test_3000.json",
         "model_path": f"{args.base_dir}/{args.run_type}4_llava_lora_ours"},
        {"question_file": "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/CLEVR/test_3000.json",
         "model_path": f"{args.base_dir}/{args.run_type}5_llava_lora_ours"},
        {"question_file": "~/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/Flickr30k/test_3000.json",
         "model_path": f"{args.base_dir}/{args.run_type}6_llava_lora_ours"},
    ]

    for idx, config in enumerate(configs, start=1):
        args.question_file = config["question_file"]
        args.model_path = config["model_path"]
        args.model_base = BASE_MODEL
        
        act = eval_model(args)
        with open(f'activation_{idx}.pkl', 'wb') as f:
            pickle.dump(act, f)

    activations = []
    for idx in range(1, len(configs) + 1):
        with open(f'activation_{idx}.pkl', 'rb') as f:
            activations.append(pickle.load(f))

    for layer in range(32):
        kernel_CKA_matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                if i <= j:
                    activation_i = activations[i]
                    activation_j = activations[j]
                    
                    kernel_result = kernel_CKA(activation_i[f'layer_{layer}_final_output'], 
                                               activation_j[f'layer_{layer}_final_output'])
                    
                    kernel_CKA_matrix[i, j] = np.round(kernel_result, 4)
                    kernel_CKA_matrix[j, i] = kernel_CKA_matrix[i, j]
                    
        print(f'layer {layer}:')
        print(kernel_CKA_matrix)
        
        plt.figure(figsize=(10, 8), dpi=500)
        sns.heatmap(kernel_CKA_matrix, annot=False, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        plt.savefig(f'kernel-CKA-layer{layer}.png')
        plt.close()
