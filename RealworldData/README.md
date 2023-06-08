Pre-Alignment and Color correction for Real-world Data 
====
****
## Datesets
* The CameraFusion (by iPhone11) dataset can be downloaded from this [link](https://github.com/Tengfei-Wang/DCSR).
* The RealMCVSR (by iPhone12) dataset can be downloaded from this [link](https://github.com/codeslake/RefVSR).
****
## Dependencies
* Color correction is based on this [repository](https://github.com/csjcai/RealSR)(Matlab>=2021a).
* Pre-alignment is based on SIFT (Python 3.7, NumPy 1.21.0, OpenCV 4.5.3).

****

## Quick Start
We take wide-angle images SR with telephoto images on iphone11 as example.

* Pre-alignment:
  ```
  cd ./PreAlignment && sh run.sh
  ```
* Color correction
    ```
    cd ./color_correction
    
    mkdir ../Data/TeleView_SIFTAlign_cor

    run ColorIuminanceDir.m
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
