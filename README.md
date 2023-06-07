#Zero-Shot Dual-Lens Super-Resolution

> Zero-Shot Dual-Lens Super-Resolution, In *CVPR* 2023.
> Ruikang Xu, Mingde Yao, Zhiwei Xiong.
> 
[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Zero-Shot_Dual-Lens_Super-Resolution_CVPR_2023_paper.pdf)|[Slides](https://cvpr.thecvf.com/media/cvpr-2023/Slides/22470.pdf)|[Video](https://youtu.be/ChHAIGyDFAI)
****
## Datesets

### Real-world Datset
* The CameraFusion (by iPhone11) dataset can be downloaded from this [link](https://github.com/Tengfei-Wang/DCSR).
* The RealMCVSR (by iPhone12) dataset can be downloaded from this [link](https://github.com/codeslake/RefVSR).


### Synthesized Datset
* The HCI_new dataset can be downloaded from this [link](https://lightfield-analysis.uni-konstanz.de/).
* The Middlebury2021 dataset can be downloaded from this [link](https://vision.middlebury.edu/stereo/data/scenes2021/).
* Other stereo image datasets ([Flick1024](https://yingqianwang.github.io/Flickr1024/), [Holopix50k](https://leiainc.github.io/holopix50k/)) and light-field datasets ([NTIRE2023](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2023), [Stanford](http://lightfields.stanford.edu/LF2016.html)) also can be used.

****

## Dependencies
* Python 3.8.8, PyTorch 1.8.0, torchvision 0.9.0.
* NumPy 1.24.2, OpenCV 4.7.0, Tensorboardx 2.5.1, Pillow, Imageio. 
****

## Data Pre-processing
### Dependencies
Please follow README files in the corresponding subfolders to bulid environments for runing.  
### Real-world Datset
We take wide-angle images SR with telephoto images on iphone11 as example.

* Pre-alignment:
  ```
  cd ./RealworldData/PreAlignment && sh run.sh
  ```
* Color correction:
  ```
  cd ./RealworldData/color_correction

  mkdir ./Data/TeleView_SIFTAlign_cor

  run ColorIuminanceDir.m
  ```

### Synthesized Datset
We take 2x downsampling using isotropic Gaussian kernel with slight JPEG compression as example.


* Synthesized Data Generation:
` cd ./SynthesizedData/DataGeneration && sh run.sh `

<br/>

* Pre-Alignment:
` cd ./SynthesizedData/PreAlignment && sh run.sh `
****

## Quick Start
### Real-world Datset
We take wide-angle images SR with telephoto images on iphone11 as example.
* Degradation‐invariant Alignment for Dual-lens Images:
  ```
  cd ./Alignment && sh align_real.sh
  ```
* ZSSR with Degradation‐aware Training and Inference:
  ```
  cd ./SR && sh sr_real.sh
  ```

### Synthesized Datset
We take 2x downsampling using isotropic Gaussian kernel with slight JPEG compression as example.
* Degradation‐invariant Alignment for Dual-lens Images:
  ```
  cd ./Alignment && sh align_synth.sh
  ```
* ZSSR with Degradation‐aware Training and Inference:
  ```
  cd ./SR && sh sr_synth.sh
  ```

****

## Contact
Any question regarding this work can be addressed to xurk@mail.ustc.edu.cn and mdyao@mail.ustc.edu.cn.

****


## Citation
If you find our work helpful, please cite the following paper.
```
@InProceedings{Xu_2023_CVPR,
    author    = {Xu, Ruikang and Yao, Mingde and Xiong, Zhiwei},
    title     = {Zero-Shot Dual-Lens Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
```