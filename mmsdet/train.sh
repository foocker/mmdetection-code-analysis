# train
config_p=/root/Codes/Synthesize/mmsdet/centernet/configs/res18_centernet.py
# python main.py ctdet --exp_id coco_dla  --arch res_18 --batch_size 16 --master_batch 4 --lr 1.25e-4  --gpus 0,1
python main.py ctdet $config_p --exp_id coco_dla  --arch resdcn_18 --batch_size 16 --master_batch 4 --lr 1.25e-4  --gpus 0,1
# python main.py --exp_id coco_dla --batch_size 16 --master_batch 4 --lr 1.25e-4  --gpus 0,1
# test
# python test.py --exp_id coco_dla --not_prefetch_test ctdet --load_model ~/Codes/CenterNet/exp/ctdet/coco_dla/model_best.pth
#demo
# python demo.py ctdet --debug 1 --demo ~/Codes/CenterNet/data/cig_box/images/train/JPEGImages --load_model ~/Codes/CenterNet/exp/ctdet/coco_dla/model_best.pth