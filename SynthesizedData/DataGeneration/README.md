#Synthesized Data Generation
****
## Datesets
* The HCI_new dataset can be downloaded from this [link](https://lightfield-analysis.uni-konstanz.de/).
* The Middlebury2021 dataset can be downloaded from this [link](https://vision.middlebury.edu/stereo/data/scenes2021/).
* Other stereo image datasets ([Flick1024](https://yingqianwang.github.io/Flickr1024/), [Holopix50k](https://leiainc.github.io/holopix50k/)) and light-field datasets ([NTIRE2023](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2023), [Stanford](http://lightfields.stanford.edu/LF2016.html)) also can be used.
****
## Dependencies
* Synthesized data generation is based on this [repository](https://github.com/assafshocher/BlindSR_dataset_generator), you can follow its requirements.
* Python 3.7, NumPy 1.16.0, Matplotlib 3.2.2, OpenCV 4.6.0, Pillow, Imageio.
****

## Quick Start
We take 2x downsampling using isotropic Gaussian kernel with slight JPEG compression as example.
You can directly run `sh run.sh` or run script files step by setp:

* Downsampling wide-angle images:

    ` python ./downSample_isoKernel.py --image_path '../Data/WideView_GT' --save_path '../Data/WideView_iso2x' --kernel_path '../Data/kernel_WideView_iso2x' `
<br/>
* Adding JPEG compression to downsampled images:

    ` python ./JPEG_compression.py --image_path '../Data/WideView_iso2x' --save_path '../Data/WideView_iso2x_JPEG75' `
<br/>
* Cropping the center area of telephoto images:

    ` python ./center_crop.py --image_path '../Data/TeleView' --save_path '../Data/TeleView_crop' `
<br/>
* Cropping the center area of downsampled wide-angle images (Only for alignment of ZeDuSR):

    ` python ./center_crop.py --image_path '../Data/WideView_iso2x_JPEG75' --save_path '../Data/WideView_iso2x_JPEG75_crop' `


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