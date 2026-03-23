import os
import re
        
llava_path = "~Documents/kienNguyen/HiDe-LLaVA/llava-7b-v1-5"
clip_path = "~Documents/kienNguyen/HiDe-LLaVA/clip-vit-large-patch14-336"
datasets_path = "~Documents/kienNguyen/HiDe-LLaVA/UCIT/datasets"
instructions_path = "~Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions"
output_path = "~/Documents/kienNguyen/HiDe-LLaVA/"

replacements = {
    "/llava-v1.5-7b" : llava_path,
    "/clip-vit-large-patch14-336" : clip_path,
    "/UCIT/datasets" : datasets_path,
    "/your_path/HiDe" : f"{output_path}/HiDe",
}
    
datasets = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"]
for ds in datasets:
    replacements[f"/UCIT/instructions/ds"] = f"{instructions_path}/{ds}"
        
# Fix the train_all.sh paths pointing to hardcoded original author directories
replacements["/mnt/haiyangguo/mywork/CL-MLLM/LLaVA-HiDe/scripts/CoIN/Train_UCIT/"] = "./scripts/HiDe/Train_UCIT/"
# Just in case it was modified before
replacements["/mnt/haiyangguo/mywork/CL-MLLM/LLaVA-HiDe/scripts/HiDe/Train_UCIT/"] = "./scripts/HiDe/Train_UCIT/"


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
    for old_str, new_str in replacements.items():
        new_content = new_content.replace(old_str, new_str)

    # This regex looks for --include localhost:X,Y,Z and removes it entirely
    # so deepspeed defaults to using whatever GPUs are available in the system
    new_content = re.sub(r'--include\s+localhost:[0-9,]+\s*', '', new_content)
        
    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated paths in: {file_path}")
        modified_count += 1
        
print(f"\nDone! Updated {modified_count} script files.")
