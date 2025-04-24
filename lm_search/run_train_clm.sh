export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=3,2,1,0 torchrun --master-port=9088 --nproc_per_node=4 train_clm.py \
                                    --trainer_class=trainer \
                                    --do_train \
                                    --model_type=gpt2 \
                                    --tokenizer_name=tokenizer_496 \
                                    --output_dir=output_test \
                                    --dataset_name=dataset \
                                    --per_device_train_batch_size=32 \
                                    \
                                    --overwrite_output_dir=True \
                                    --logging_steps=100 \
                                    --num_train_epochs=20000 \
                                    --remove_unused_columns=False \
                                    --learning_rate=5e-5 \
                                    --save_steps=1000 \
                                    --lr_scheduler_type=constant \
                                    \
                                    --fp16 \
                                    \
                                    | tee output_test/test.log
                                    # --model_name_or_path=clm_gen_v100/checkpoint-56000
                                    # --resume_from_checkpoint=clm_gen/checkpoint-52000

                                    
                                    # --do_eval \
                                    # --per_device_eval_batch_size=128 \
                                    # --evaluation_strategy=steps \
                                    # --eval_steps=80000000000000 \