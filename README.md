# Spiner: *Sp*iral gangl*i*on *n*euron profil*er*
Designed for automated detection of spiral ganglion neurons in 3D image datasets of cleared cochleae.
[fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet) (Python) is adopted for cell detection.  

## Build a web interface on your own
[Source codes](https://github.com/reubenrosenNCSU/cellannotation) for building a web interface which allows users to test, annotate, fine-tune, and adapt the pre-trained SGN object detection model.

## Local installation with Anaconda (Windows)
Download this repository.  
*Check [here](https://docs.anaconda.com/anaconda/install/index.html) for Anaconda documentation.*  
Open Anaconda Prompt and create an virtual environment - the package versions below are not mandontary but should work.  
`conda create -n spiner-env python=3.7 tensorflow-gpu=2.3 tensorflow=2.3=mkl_py37h936c3e2_0 keras=2.4 numpy=1.19.2 matplotlib=3.3.4 pyqtwebengine=5.15.7 spyder=5.2.2`      
Activate the environment: `conda activate spiner-env`.  

Go to the code directory, e.g. `.../Spiner/`, in Anaconda Prompt.  
Install keras-retinanet under the Spiner directory by running `pip install . --user`.  
Run `python setup.py build_ext --inplace` to compile Cython code.

## RetianNet training and testing
Please refer to [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet) for more details, which includes information about how to train the model on a custom dataset and the annotation format of a custom CSV dataset. 

For training on a custom CSV dataset:
```shell
# Running directly from the repository in Anaconda Prompt
python ./keras_retinanet/bin/train.py csv /path/to/csv/file/containing/annotations/training /path/to/csv/file/containing/classes --val-annotations /path/to/csv/file/containing/annotations/validation
```
Check `./keras_retinanet/bin/train.py` for more details about hyperparameters such as initial weights, learning rate, batch size, etc. 

## Cell detection
Follow `inferenceStitch.py`, an automated cell detection and stitching pipeline.  
Please go through all the parameters in the "Input parameters" section before running the script. 

 - Input:
     - A single-tile dataset or a multi-tile dataset structured as [a two-level hierarchy of folders](https://github.com/abria/TeraStitcher/wiki/Supported-volume-formats#two-level-hierarchy-of-folders).  
 - Output:  
     - List of predictions (.csv) per tile *(required for **Cell stitching** if multi-tile)*.  
     - Images with predictd boxes (.png).
     - List of cell centroids (.csv).

### Stitching (Optional)
Software: [TeraStitcher](https://abria.github.io/TeraStitcher/)  
 - Input:
     - A multi-tile dataset structured as [a two-level hierarchy of folders](https://github.com/abria/TeraStitcher/wiki/Supported-volume-formats#two-level-hierarchy-of-folders).
 - Output:  
     - XML descriptor (.xml) containing tile positions *(required for **Cell stitching**)*.  

### Cell stitching (Optional)

 - Input:
     - List of predictions (.csv) per tile *(from **Cell detection**)*.  
     - XML descriptor (.xml) containing tile positions *(from **Stitching**)*.  
 - Output:  
     - List of stitched cell centroids (.csv).  

# Contact
ycai23@ncsu.edu; rene.cai@unc.edu

# Reference
- https://github.com/fizyr/keras-retinanet
- Bria, Alessandro, and Giulio Iannello. "TeraStitcher-a tool for fast automatic 3D-stitching of teravoxel-sized microscopy images." *BMC bioinformatics* 13.1 (2012): 1-15.  
