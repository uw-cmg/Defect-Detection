# Defect-detection
Updated: March 20 2018

Loop defect detection packages using Faster-RCNN / SSD and local
image analysis methods.

Automated image analysis for open elliptical loop 
dislocation in STEM images of irradiated alloys


## Dependency
Python version >= 3.5 
* [ChainerCV](http://chainercv.readthedocs.io/en/latest/index.html)
* [Chainer](https://github.com/chainer/chainer)
* [OpenCV](https://opencv.org/)
* [Scikit-image](http://scikit-image.org/)

## Installation
Install miniconda (or Anaconda) for python 3

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda config --set always_yes yes --set changeps1 no
conda update -q conda
```
Download ChainerCV and go to the root directory of ChainerCV
```
git clone https://github.com/chainer/chainercv
cd chainercv
conda env create -f environment.yml
source activate chainercv
```
Install ChainerCV
```
pip install -e .
```
Install scikit-images
```
conda install scikit-image
```
Download the defect detection code package and go to the directory of defect detection
```
cd ..
git clone https://github.com/leewaymay/defect-detection.git
cd defect-detection
```
## Usage
### Data
Download data at [Here](https://www.dropbox.com/sh/ttl5u14uzqxrili/AAAa1XMxP9AVJPQ3ie7xZZVxa?dl=0). Remember the ```PATH``` of the data root directory.

### Train
To train the model on the dataset, run
```python sample_train.py```
### On-going Development
See the ongoing development of the project, see jupyter notebook ```Defect detection debug.ipynb```
