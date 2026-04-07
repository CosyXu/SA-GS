<p align="center">
<h1 align="center">&nbsp;SA-GS: Season-Aware Affine 3D Gaussian Splatting for Satellite Image Rendering</h1>
  <p align="center">
    Yihang Xu · Qiulei Dong
  </p>
  <div align="center"></div>
</p>
<p align="justify">
We introduce SA-GS, a Season-aware Affine 3D Gaussian Splatting method for rendering 
line-array satellite images under the affine camera model. Unlike existing 3DGS-based methods 
that are explored under the pinhole camera model, SA-GS explores an affine version of 3DGS 
under the affine camera model for effective rendering of line-array satellite 
images, incorporates a seasonal embedding module to tolerate significant textural variations 
among cross-season satellite images, and introduces an affine view augmentation module to 
generate synthetic satellite images as additional training supervision for further improving 
the rendering performance.
</p>

## Installation
Clone the repository and create a `python=3.11` Anaconda environment with CUDA toolkit 11.8 installed using
```bash
git clone https://github.com/CosyXu/SA-GS.git
cd SA-GS

conda create -y -n sa python=3.11
conda activate sa
conda install -y --override-channels -c nvidia/label/cuda-11.8.0 cuda-toolkit
conda env config vars set NERFBASELINES_BACKEND=python
pip install --upgrade pip
pip install -r requirements.txt
pip install nerfbaselines>=1.2.0
pip install ./submodules/diff-gaussian-rasterization/ --no-build-isolation
pip install ./submodules/simple-knn/ --no-build-isolation
pip install -e .
```

## Dataset
In order to train/evaluate on the DFC2019 dataset, please download the processed version from the following link:
[Google Drive](https://drive.google.com/drive/folders/168x4aUmSh-8AvYO_DZ_xccNXgELlLiKN?usp=drive_link)

## Training
To start the training on the DFC2019 dataset, run the following command:
```bash
sa train --data {path to data} --output {path to output}
```

## Evaluation
To reproduce the quantitative results reported in the paper, run the following commands:
```bash
tar -zxvf predictions-{iters}.tar.gz
python metrics.py -m {path to output}
```

## Acknowledgement
Parts of this repository are derived from [WildGaussians](https://github.com/jkulhanek/wild-gaussians), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), and [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/). We would like to express our gratitude to the authors of these works.