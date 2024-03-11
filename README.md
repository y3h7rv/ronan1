For ADM(guided-diffusion), here are the setup commands. Use the `classifier_script` file in `scripts` folder to start
1. command ```SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing 100"``` into cmd. (With 48G cuda memory, the maximum timestep is 150)
2. command `MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"`
3. `python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path ../models/64x64_classifier.pt --classifier_depth 4 --model_path ../models/64x64_diffusion.pt $SAMPLE_FLAGS`
