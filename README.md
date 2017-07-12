NOTE: See the original Readme for py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn). The description below adds specific details about running faster-rcnn on Caltech dataset.

### Training and Testing
1. Create following directory structure:
 	```Shell
 	mkdir caltech; cd caltech;
 	mkdir unzip
 	mkdir data
 	mkdir data/JPEGImages
 	mkdir data/ImageSets
	```
	
2. Download the video sequences and annotations

	```Shell
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set00.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set01.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set02.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set03.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set04.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set05.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set06.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set07.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set08.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set09.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/set10.tar
	wget https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/annotations.zip
	```
	
3. Extract all of these tars into a temporary directory `unzip`
	```Shell
	tar -xf downloaded/set00.tar --directory unzip/
	tar -xf downloaded/set01.tar --directory unzip/
	tar -xf downloaded/set02.tar --directory unzip/
	tar -xf downloaded/set03.tar --directory unzip/
	tar -xf downloaded/set04.tar --directory unzip/
	tar -xf downloaded/set05.tar --directory unzip/
	tar -xf downloaded/set06.tar --directory unzip/
	tar -xf downloaded/set07.tar --directory unzip/
	tar -xf downloaded/set08.tar --directory unzip/
	tar -xf downloaded/set09.tar --directory unzip/
	tar -xf downloaded/set10.tar --directory unzip/
	```
	
4. Move inside `data` directory
	```Shell
	cd data
	```	    
	
5. Extract the `annotations.zip` file
	```Shell
	unzip ../annotations.zip .
	```	
	This should create a new directory `annotations` inside the `data` directory.
	
6. Parse and extract all the images from dataset using this [utility](https://github.com/govindnh4cl/caltech-pedestrian-dataset-converter).
Update source (`caltech`) and destination (`caltech/data/JPEGImages`) directory paths inside the scripts and execute:
	```Shell
	python convert_seqs.py
	```	
	This should create multiple sets directories inside `JPEGImages`. Each containing multiple `Vxxx` directories which in-turn containing multiple `.jpg` images.
	
7. Create ImageSets
Image sets are the text files containing the image names for train and test sets. There are two ways to achieve this:

    A: Directly downloaded the `train_1x.txt` and `test_1x.txt` from [here](https://github.com/govindnh4cl/caltech-pedestrian-dataset-converter/tree/master/dump/ImageSets).

    B: Or, create them from scratch by editing the `convert_seqs.py` with the following changes
    1. Set `print_names = 1`
    2. Edit the `sets` list for respective set names to be included.
    3. Set `interval` based on whether 1x or 10x set is to be generated for caltech dataset.
    Example:
    
    	Generation of 1x training set:
    	(`print_names = 1; sets = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']; interval = 30;`)
    	```Shell
      	$python convert_seqs.py > ImageSets/train_1x.txt
    	```
    
        Generation of 10x training set(optional):
    	(`print_names = 1; sets = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']; interval = 3;`)
    	```Shell
      	$python convert_seqs.py > ImageSets/train_10x.txt
    	```	
    	
        Generation of 1x test set:
    	(`print_names = 1; sets = ['set06', 'set07', 'set08','set09', 'set10']; interval = 30;`)
    	```Shell
      	$python convert_seqs.py > ImageSets/test_1x.txt	
    	```
        
    The `unzip` directory can be deleted now.
	
8. At this point, you should have to following directory structure
	```Shell
  	$caltech/data/
  	$caltech/data/annotations/
	$caltechdata/annotations/set00/
	$caltech/data/annotations/set01/
	$...
	$caltech/data/annotations/set10/
  	$caltech/data/JPEGImages
	$caltech/data/JPEGImages/set00
	$caltech/data/JPEGImages/set01
	$...
	$caltech/data/JPEGImages/set10
  	$caltech/data/ImageSets/train_1x.txt
	$caltech/data/ImageSets/test_1x.txt
    ```
	
9. If the directory that you cloned Faster R-CNN is FRCN_ROOT, then create symlinks inside it for the dataset
	```Shell
    cd $FRCN_ROOT/data
    ln -s path/to/caltech caltech
    ```
    Using symlinks is a good idea because you will likely want to share the same caltech dataset installation between multiple projects.

### Download pre-trained ImageNet models

Pre-trained ImageNet models should be downloaded for the VGG16 network described in the faster-RCNN paper (ZF may not be work as of now).

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

**Alt-opt method not supported, using end2end**

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [--set ...]
# GPU_ID is the GPU you want to train on
# --set ... allows you to specify fast_rcnn.config options, e.g.
# --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```

**The results are produced as Average Precision and not as Miss-Rate at present.**