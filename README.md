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

## Usage
### Data
Download data at [Here](https://www.dropbox.com/sh/ttl5u14uzqxrili/AAAa1XMxP9AVJPQ3ie7xZZVxa?dl=0)

Change the data root dir to your data folder path in ```./utils.DefectDetectionDataset.py```
### Train
To train the model on the dataset, run
```python sample_train.py```
### On-going Development
See the ongoing development of the project, see jupyter notebook *Defect detection debug.ipynb*