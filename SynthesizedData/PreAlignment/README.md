#Pre-Alignment for Synthesized Data 
****
## Datesets
* The HCI_new dataset can be downloaded from this [link](https://lightfield-analysis.uni-konstanz.de/).
* The Middlebury2021 dataset can be downloaded from this [link](https://vision.middlebury.edu/stereo/data/scenes2021/).
* Other stereo image datasets ([Flick1024](https://yingqianwang.github.io/Flickr1024/), [Holopix50k](https://leiainc.github.io/holopix50k/)) and light-field datasets ([NTIRE2023](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2023), [Stanford](http://lightfields.stanford.edu/LF2016.html)) also can be used.
****
## Dependencies
* Pre-alignment is based on SIFT (Scale Invariant Feature Transform).
* Python 3.7, NumPy 1.21.0, OpenCV 4.5.3.
****

## Quick Start
We take 2x downsampling using isotropic Gaussian kernel with slight JPEG compression as example.
You can run `sh run.sh` or directly run the script file:

``` 
python ./sift_align.py --LR_dir '../Data/WideView_iso2x_JPEG75_crop' --Ref_dir '../Data/TeleView_crop' --save_path '../Data/TeleView_crop_SIFTAlign' --scale 2 
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