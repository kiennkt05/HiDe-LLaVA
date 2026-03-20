#!/bin/bash
BASE_DIR="/home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA"

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh hide-task3 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/Task3_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh hide-task3 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/Task3_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_vizwiz.sh hide-task3 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/Task3_llava_lora_ours 0

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh hide-task2 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/Task2_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh hide-task2 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/Task2_llava_lora_ours 0

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh hide-task1 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/Task1_llava_lora_ours 0

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh ab-task3 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/ab_Task3_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh ab-task3 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/ab_Task3_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_vizwiz.sh ab-task3 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/ab_Task3_llava_lora_ours 0

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh ab-task2 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/ab_Task2_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh ab-task2 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/ab_Task2_llava_lora_ours 0

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh a-task3 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/a_Task3_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh a-task3 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/a_Task3_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_vizwiz.sh a-task3 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/a_Task3_llava_lora_ours 0

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh a-task2 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/a_Task2_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh a-task2 /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/HiDe/a_Task2_llava_lora_ours 0