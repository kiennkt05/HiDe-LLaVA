import os
import argparse
import re

def main():
    parser = argparse.ArgumentParser(description="Update paths in HiDe-LLaVA shell scripts robustly.")
    parser.add_argument("--base_path", type=str, help="Absolute path to your HiDe-LLaVA directory")
    args = parser.parse_args()
    
    if args.base_path:
        base = os.path.abspath(os.path.expanduser(args.base_path)).replace("\\", "/")
    else:
        print("=== HiDe-LLaVA Script Path Updater ===")
        user_input = input("Enter the absolute base path to your HiDe-LLaVA folder (e.g. /home/user/HiDe-LLaVA): ")
        if not user_input.strip():
            print("Base path required. Exiting.")
            return
        base = os.path.abspath(os.path.expanduser(user_input.strip())).replace("\\", "/")

    llava_path = f"{base}/llava-7b-v1-5" # Account for user's specific folder name llava-7b-v1-5
    # If the user has a llava-v1.5-7b folder instead, we account for that too:
    if not os.path.exists(llava_path) and os.path.exists(f"{base}/llava-v1.5-7b"):
        llava_path = f"{base}/llava-v1.5-7b"

    clip_path = f"{base}/clip-vit-large-patch14-336"
    datasets_path = f"{base}/UCIT/datasets"
    instructions_path = f"{base}/UCIT/instructions"
    
    # We will use Regex to confidently overwrite the argument flags in the scripts
    # so it doesn't matter what broken path is currently saved in them.
    replacements = [
        (r'--model_name_or_path\s+[^\s\\]+', f'--model_name_or_path {llava_path}'),
        (r'--pretrain_mm_mlp_adapter\s+[^\s\\]+', f'--pretrain_mm_mlp_adapter {llava_path}/mm_projector.bin'),
        (r'--image_folder\s+[^\s\\]+', f'--image_folder {datasets_path}'),
        (r'--vision_tower\s+[^\s\\]+', f'--vision_tower {clip_path}'),
        (r'--text_tower\s+[^\s\\]+', f'--text_tower {clip_path}'),
        (r'--model-base\s+[^\s\\]+', f'--model-base {llava_path}'),
        (r'--image-folder\s+[^\s\\]+', f'--image-folder {datasets_path}'),
        (r'--text-tower\s+[^\s\\]+', f'--text-tower {clip_path}'),
        (r'--include\s+localhost:[0-9,]+\s*', '')  # DeepSpeed GPU cleanup
    ]

    # Dynamically match datasets for data_path
    datasets = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]
    def build_data_path_replacement(match):
        for ds in datasets:
            if ds in match.group():
                return f"--data_path {instructions_path}/{ds}/train.json"
        return match.group()

    def build_test_data_path_replacement(match):
        for ds in datasets:
            if ds in match.group():
                return f"--question-file {instructions_path}/{ds}/test_3000.json"
        return match.group()

    script_files = []
    for root, dirs, files in os.walk("scripts"):
        for file in files:
            if file.endswith(".sh"):
                script_files.append(os.path.join(root, file))
                
    modified_count = 0
    for file_path in script_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        new_content = content
        
        # Apply strict regex replacements based on flags
        for pattern, replacement in replacements:
            new_content = re.sub(pattern, replacement, new_content)

        # Apply dataset specific replacements for --data_path and --question-file
        new_content = re.sub(r'--data_path\s+[^\s\\]+', build_data_path_replacement, new_content)
        new_content = re.sub(r'--question-file\s+[^\s\\]+', build_test_data_path_replacement, new_content)
        new_content = re.sub(r'--annotation-file\s+[^\s\\]+', build_test_data_path_replacement, new_content)

        # Fix output_dir which is unique per script depending on the task name
        # We find the output_dir and replace the base part of the path, preserving the end
        def fix_output_dir(match):
            old_path = match.group(1)
            # Find the /HiDe/ directory structure and preserve it
            if "HiDe/" in old_path:
                suffix = old_path.split("HiDe/")[-1]
            elif "Task" in old_path:
                suffix = old_path.split("/")[-1]
            else:
                return match.group(0)
            return f"--output_dir {base}/HiDe/{suffix}"
            
        new_content = re.sub(r'--output_dir\s+([^\s\\]+)', fix_output_dir, new_content)
        
        # Eval output directories:
        def fix_eval_output_dir(match):
            old_path = match.group(1)
            if "results/UCIT" in old_path:
                suffix = old_path.split("results/UCIT/")[-1]
            else:
                return match.group(0)
            return f"--answers-file {base}/results/UCIT/{suffix}"

        def fix_eval_result_dir(match):
            old_path = match.group(1)
            if "results/UCIT" in old_path:
                suffix = old_path.split("results/UCIT/")[-1]
            else:
                return match.group(0)
            return f"--output-dir {base}/results/UCIT/{suffix}"

        new_content = re.sub(r'--answers-file\s+([^\s\\]+)', fix_eval_output_dir, new_content)
        new_content = re.sub(r'--output-dir\s+([^\s\\]+)', fix_eval_result_dir, new_content)

        # Model Paths for eval
        def fix_eval_model_path(match):
            old_path = match.group(1)
            if "HiDe/" in old_path:
                suffix = old_path.split("HiDe/")[-1]
            elif "Task" in old_path:
                suffix = old_path.split("/")[-1]
            else:
                return match.group(0)
            return f"--model-path {base}/HiDe/{suffix}"

        new_content = re.sub(r'--model-path\s+([^\s\\]+)', fix_eval_model_path, new_content)

        # Hardcoded fix for script calling directories
        new_content = new_content.replace("/mnt/haiyangguo/mywork/CL-MLLM/LLaVA-HiDe/scripts/HiDe/Train_UCIT/", "./scripts/HiDe/Train_UCIT/")
        new_content = new_content.replace("/mnt/haiyangguo/mywork/CL-MLLM/LLaVA-HiDe/scripts/CoIN/Train_UCIT/", "./scripts/HiDe/Train_UCIT/")
        
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated paths in: {file_path}")
            modified_count += 1
            
    print(f"\nDone! Updated {modified_count} script files.")

if __name__ == "__main__":
    main()
