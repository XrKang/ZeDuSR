#Pre-Alignment for Real-world Data 
****
## Datesets
* The CameraFusion (by iPhone11) dataset can be downloaded from this [link](https://github.com/Tengfei-Wang/DCSR).
* The RealMCVSR (by iPhone12) dataset can be downloaded from this [link](https://github.com/codeslake/RefVSR).
****
## Dependencies
* Pre-alignment is based on SIFT (Scale Invariant Feature Transform).
* Python 3.7, NumPy 1.21.0, OpenCV 4.5.3.
****

## Quick Start
We take wide-angle images SR with telephoto images on iphone11 as example.
You can run `sh run.sh` or directly run the script file:
```
python ./sift_align.py --mode 'iphone11_wideSRTele' --wide_dir '../Data/WideView' --tele_dir '../Data/TeleView' --Tele_savePath '../Data/TeleView_SIFTAlign' --WideCrop_savePath '../Data/WideView_crop' 
```

****

##Citation
If you find our work helpful, please cite the following paper.
```
@InProceedings{Xu_2023_CVPR,
    author    = {Xu, Ruikang and Yao, Mingde and Xiong, Zhiwei},
    title     = {Zero-Shot Dual-Lens Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
```