python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/decoder_uv.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/decoder_uv.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/decoder_y.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/decoder_y.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/hyper_decoder_uv.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/hyper_decoder_uv.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/hyper_decoder_y.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/hyper_decoder_y.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/hyper_scale_decoder_uv.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/hyper_scale_decoder_uv.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/hyper_scale_decoder_y.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/decoder/hyper_scale_decoder_y.yml