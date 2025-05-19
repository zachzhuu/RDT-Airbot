python -m scripts.airbot_inference \
    --max_publish_step=750 \
    --use_actions_interpolation \
    --pretrained_model_name_or_path="checkpoints/checkpoint-150000/pytorch_model/mp_rank_00_model_states.pt" \
    --lang_embeddings_path="lang/catch_cup_gray.pt" \
    --ctrl_freq=25    # your control frequency
