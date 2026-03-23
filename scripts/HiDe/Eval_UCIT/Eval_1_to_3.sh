BASE_DIR="/home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA"

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh hide-task3 $BASE_DIR/HiDe/Task3_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh hide-task3 $BASE_DIR/HiDe/Task3_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_vizwiz.sh hide-task3 $BASE_DIR/HiDe/Task3_llava_lora_ours 0

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh hide-task2 $BASE_DIR/HiDe/Task2_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh hide-task2 $BASE_DIR/HiDe/Task2_llava_lora_ours 0

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh hide-task1 $BASE_DIR/HiDe/Task1_llava_lora_ours 0
