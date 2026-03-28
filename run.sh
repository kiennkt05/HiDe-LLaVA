sh ./scripts/HiDe/Train_UCIT/Task6.sh

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh hide-task6 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/Task6_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh hide-task6 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/Task6_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_vizwiz.sh hide-task6 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/Task6_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_iconqa.sh hide-task6 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/Task6_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_clevr.sh hide-task6 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/Task6_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_flickr30k.sh hide-task6 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/Task6_llava_lora_ours 0

sh ./scripts/HiDe/Train_UCIT/a_Task2.sh

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh a-task2 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/a_Task2_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh a-task2 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/a_Task2_llava_lora_ours 0

sh ./scripts/HiDe/Train_UCIT/a_Task3.sh

sh ./scripts/HiDe/Eval_UCIT/eval_imagenet.sh a-task3 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/a_Task3_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_arxivqa.sh a-task3 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/a_Task3_llava_lora_ours 0
sh ./scripts/HiDe/Eval_UCIT/eval_vizwiz.sh a-task3 ~/Documents/kienNguyen/HiDe-LLaVA//HiDe/a_Task3_llava_lora_ours 0
