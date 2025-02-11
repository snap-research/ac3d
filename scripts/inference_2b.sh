MODEL_PATH="THUDM/CogVideoX-2b"

video_root_dir="/path/to/dataset" # subfolders: annotations/ pose_files/ video_clips/
annotation_json="annotations/test.json"

dir=`pwd`
ckpt_dir=${dir}/out/2B
ckpt_steps=10000
ckpt_file=checkpoint-${ckpt_steps}.pt

out_dir=${ckpt_dir}/test/${ckpt_steps}
ckpt_path=${ckpt_dir}/${ckpt_file}
output_path=${out_dir}

python inference/cli_demo_camera.py \
    --video_root_dir $video_root_dir \
    --annotation_json $annotation_json \
    --prompt "Three fluffy sheep sit side by side at a rustic wooden table, each eagerly digging into their bowls of spaghetti. The pasta is tangled playfully around their woolly faces, and the bright red sauce splatters across their fur. The scene takes place in a lush, green meadow surrounded by rolling hills, with a few grazing cows in the background." \
    --base_model_path $MODEL_PATH \
    --controlnet_model_path $ckpt_path \
    --output_path $output_path \
    --start_camera_idx 0 \
    --end_camera_idx 7 \
    --stride_min 2 \
    --stride_max 2 \
    --controlnet_weights 1.0 \
    --controlnet_guidance_start 0.0 \
    --controlnet_guidance_end 0.4 \
    --controlnet_transformer_num_attn_heads 4 \
    --controlnet_transformer_attention_head_dim 32 \
    --controlnet_transformer_out_proj_dim_factor 64 \
    --controlnet_transformer_out_proj_dim_zero_init