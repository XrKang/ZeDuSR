#Synthesized Data Generation and Pre-Alignment
****
## Datesets
* The HCI_new dataset can be downloaded from this [link](https://lightfield-analysis.uni-konstanz.de/).
* The Middlebury2021 dataset can be downloaded from this [link](https://vision.middlebury.edu/stereo/data/scenes2021/).
* Other Stereo Image Dataset ([Flick1024](https://yingqianwang.github.io/Flickr1024/), [Holopix50k](https://leiainc.github.io/holopix50k/)) or LightField Dataset ([NTIRE2023](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LF-Image-SR/tree/NTIRE2023), [Stanford](http://lightfields.stanford.edu/LF2016.html))
****
## Dependencies

* Synthesized data generation is based on this [repository](https://github.com/assafshocher/BlindSR_dataset_generator), you can follow its requirements (Python 3.7, NumPy 1.16.0, Matplotlib 3.2.2, OpenCV 4.6.0, Pillow, Imageio).
* Pre-alignment is based on SIFT (Python 3.7, NumPy 1.21.0, OpenCV 4.5.3.).

****

## Quick Start
We take 2x downsampling using isotropic Gaussian kernel with slight JPEG compression as example.


* Synthesized Data Generation:
` cd ./DataGeneration && sh run.sh `

<br/>

* Pre-Alignment:
` cd ./PreAlignment && sh run.sh `



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