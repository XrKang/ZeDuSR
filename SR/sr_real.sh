python ./ZSSR_real.py \
        --scale 2 \
        --shave 4 \
        --train_lr '../RealworldData/Data/DIAlign/iPhone11_wideSRTele/IMG_2936/out_1000_warp.png' \
        --train_hr '../RealworldData/Data/DIAlign/iPhone11_wideSRTele/IMG_2936/HR.png' \
        --test_lr '../RealworldData/Data/WideView/IMG_2936.jpg' \
        --test_hr '../RealworldData/Data/WideView/IMG_2936.jpg' \
        --Invari_map '../RealworldData/Data/DIAlign/iPhone11_wideSRTele/IMG_2936/PatchDisOut.npy' \
        --output_path './Results_Real/' \
        --dataset 'iPhone11_wideSRTele/IMG_2936'