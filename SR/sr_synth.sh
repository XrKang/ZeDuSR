python ./ZSSR.py \
        --scale 2 \
        --shave 4 \
        --train_lr '../SynthesizedData/Data/DIAlign/LF_isoJPEG2x/LF_bedroom/out_1000_warp.png' \
        --train_hr '../SynthesizedData/Data/DIAlign/LF_isoJPEG2x/LF_bedroom/HR.png' \
        --test_lr '../SynthesizedData/Data/WideView_iso2x_JPEG75/LF_bedroom.jpg' \
        --test_hr '../SynthesizedData/Data/WideView_GT/LF_bedroom.png' \
        --Invari_map '../SynthesizedData/Data/DIAlign/LF_isoJPEG2x/LF_bedroom/PatchDisOut.npy' \
        --output_path './Results_Synthesized/' \
        --dataset 'LF_isoJPEG2x/LF_bedroom'