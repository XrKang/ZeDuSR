python alignment.py \
    --input_hr '../RealworldData/Data/TeleView_SIFTAlign_cor/IMG_2936.jpg' \
    --input_lr '../RealworldData/Data/WideView_crop/IMG_2936.jpg' \
    --output_path '../RealworldData/Data/DIAlign/'\
    --dataset 'iPhone11_wideSRTele/IMG_2936' \
    --shave 5 \
    --scale 2 \
    --epochs 1001 \
    --fre_epoch 700 
