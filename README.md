# Neural Rephotographyy | MSCV '22 Capstone

Team members: Sri Nitchith, [Rohan Chacko](https://github.com/RohanChacko)    
Advisor: Prof. Aswin Sankaranarayanan

## Installation

### Tensorflow 2.4+cuda 11.0:
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


### Alternate way (from original repo)

```
# Clone the repo
git clone https://github.com/nitchith/Neural-Rephotography.git; cd Neural-Rephotography
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

```
conda create --name ${env_name} python=3.6.13
conda activate ${env_name}
conda install -c conda-forge cudatoolkit-dev=11.0 # Make sure nvidia drivers support cuda-11.0
pip install -r requirements.txt
python -m pip install tensorflow==2.4.0
python -m pip install --upgrade jax jaxlib==0.1.69+cuda110 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
### Data

We will publicly release the lego focus-aperture dataset soon.


### OOM errors
You may need to reduce the batch size to avoid out of memory errors. For example the model can be run on a NVIDIA 3080 (10Gb) using the following flag. 
```
--gin_param="Config.batch_size = 1024"
```


### Acknowledgements

This codebase is built on top of the [Mip-NeRF](https://github.com/google/mipnerf) codebase. We thank the authors for open-sourcing their work. We also thank [Neha Boloor](https://github.com/neha-boloor) for her early contributions in this project. 
