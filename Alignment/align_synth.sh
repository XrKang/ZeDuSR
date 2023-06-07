python alignment.py \
    --input_hr '../SynthesizedData/Data/TeleView_crop_SIFTAlign/LF_bedroom.png' \
    --input_lr '../SynthesizedData/Data/WideView_iso2x_JPEG75_crop/LF_bedroom.jpg' \
    --output_path '../SynthesizedData/Data/DIAlign/' \
    --dataset 'LF_isoJPEG2x/LF_bedroom' \
    --shave 5 \
    --scale 2 \
    --epochs 1001 \
    --fre_epoch 600 
