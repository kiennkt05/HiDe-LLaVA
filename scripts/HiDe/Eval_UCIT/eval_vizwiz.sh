#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=1
IDX=0

if [ ! -n "$1" ] ;then
    STAGE='hide'
else
    STAGE=$1
fi

MODELPATH=$2

if [ ! -n "$3" ] ;then
    GPU=0
else
    GPU=$3
fi

RESULT_DIR="./results/UCIT/each_dataset/VizWiz"

# for IDX in $(seq 0 $((CHUNKS-1))); do
CUDA_VISIBLE_DEVICES=$GPU python -m llava.eval.model_answer \
    --model-path $MODELPATH \
    --model-base /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/llava-7b-v1-5 \
    --question-file /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/VizWiz/test_3000.json \
    --image-folder /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/UCIT/datasets \
    --text-tower /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/clip-vit-large-patch14-336 \
    --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_caption \
    --annotation-file /home/s24gbn1/Documents/kienNguyen/HiDe-LLaVA/UCIT/instructions/VizWiz/val_coco_type_3000.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \
