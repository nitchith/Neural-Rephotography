# NeReFocus

### FA Stack - Lego truck data
```
https://drive.google.com/file/d/1ZB77OZvxHiJPli18OU_g4iZJpfUibcVT/view?usp=sharing
```


## Installation
We recommend using [Anaconda](https://www.anaconda.com/products/individual) to set
up the environment. Run the following commands:

```
# Clone the repo
git clone https://github.com/google/mipnerf.git; cd mipnerf
# Create a conda environment, note you can use python 3.6-3.8 as
# one of the dependencies (TensorFlow) hasn't supported python 3.9 yet.
conda create --name mipnerf python=3.6.13; conda activate mipnerf
# Prepare pip
conda install pip; pip install --upgrade pip
# Install requirements
pip install -r requirements.txt
```
[Optional] Install GPU and TPU support for Jax
```
# Remember to change cuda101 to your CUDA version, e.g. cuda110 for CUDA 11.0.
pip install --upgrade jax jaxlib==0.1.65+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

For Cuda 11:
```
# Set cuda-11.0 path
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Create environment
conda create --name nerefocus python=3.6.13
conda activate nerefocus

# Prepare pip
conda install pip; pip install --upgrade pip

# Install packages in requirements.txt
pip install -r requirements.txt

# Check compatible tensorflow and cudnn version to your cuda version - https://www.tensorflow.org/install/source#gpu
python -m pip install tensorflow==2.4.0
conda install cudnn=8

# Check compatible jaxlib version - https://storage.googleapis.com/jax-releases/jax_releases.html
python -m pip install --upgrade jax jaxlib==0.1.69+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html

```


## Data

Then, you'll need to download the datasets
from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download and unzip `nerf_synthetic.zip` and `nerf_llff_data.zip`.

### Generate multiscale dataset
You can generate the multiscale dataset used in the paper by running the following command,
```
python scripts/convert_blender_data.py --blenderdir /nerf_synthetic --outdir /multiscale
```

## Running

Example scripts for training mip-NeRF on individual scenes from the three
datasets used in the paper can be found in `scripts/`. You'll need to change
the paths to point to wherever the datasets are located.
[Gin](https://github.com/google/gin-config) configuration files for our model
and some ablations can be found in `configs/`.
An example script for evaluating on the test set of each scene can be found
in `scripts/`, after which you can use `scripts/summarize.ipynb` to produce
error metrics across all scenes in the same format as was used in tables in the
paper.

### OOM errors
You may need to reduce the batch size to avoid out of memory errors. For example the model can be run on a NVIDIA 3080 (10Gb) using the following flag. 
```
--gin_param="Config.batch_size = 1024"
```

## Citation
If you use this software package, please cite our paper:

```
@misc{barron2021mipnerf,
      title={Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields},
      author={Jonathan T. Barron and Ben Mildenhall and Matthew Tancik and Peter Hedman and Ricardo Martin-Brualla and Pratul P. Srinivasan},
      year={2021},
      eprint={2103.13415},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
Thanks to [Boyang Deng](https://boyangdeng.com/) for JaxNeRF.
