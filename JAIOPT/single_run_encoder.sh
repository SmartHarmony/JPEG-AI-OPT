python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_decoder_uv.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_decoder_uv.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_decoder_y.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_decoder_y.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_encoder_uv.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_encoder_uv.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_encoder_y.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_encoder_y.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_scale_decoder_uv.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_scale_decoder_uv.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_scale_decoder_y.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/hyper_scale_decoder_y.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/single_encode_uv.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/single_encode_uv.yml

python ./lothar/controller.py convert --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/single_encode_y.yml
python ./lothar/controller.py run --validate --warmup_round=5 --config=/home/huzq85/2-working/JPEG-AI-OPT/Verification_Models/encoder/single_encode_y.yml
