Title: A Novel Incremental Defect Detection Method via Elastic Heterogeneous Distillation Network

### Requirements
- Linux or macOS
- Python >= 3.6
- PyTorch 1.7
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, needed by demo and visualization
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install fvcore==0.1.1.dev200512`
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
- GCC >= 4.9``


### Build Detectron2

After having the above dependencies, run:
```
cd iOD
python setup.py build develop

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

# or, as an alternative to `setup.py`, do
# pip install [--editable] .
```
Note: you may need to rebuild detectron2 after reinstalling a different build of PyTorch.

### Running

'''
Run ..tools/train_source.py 
and adjust the .yaml file for different increments.
'''
