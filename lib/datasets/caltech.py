# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from caltech_utils import caltech_eval, parse_caltech_annotations
from fast_rcnn.config import cfg

class caltech(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        #govind: values of image_set is "train", "test" or "val"
        #govind: Ignoring year for now
        #govind: devkit_path is path to dataset directory
    
        imdb.__init__(self, 'caltech_' + image_set)
        self._year = str(year)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        #govind: Ignoring the people class. So, num_classes = 2
        self._classes = ('__background__', # always index 0
                         'person') 
        
        #self._classes = ('__background__', # always index 0
        #                 'aeroplane', 'bicycle', 'bird', 'boat',
        #                 'bottle', 'bus', 'car', 'cat', 'chair',
        #                 'cow', 'diningtable', 'dog', 'horse',
        #                 'motorbike', 'person', 'pottedplant',
        #                 'sheep', 'sofa', 'train', 'tvmonitor')        
        
        #govind: num_classes is set based on the number of classes in _classes tuple
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        #govind: self._image_index is a list of all image names for current _image_set
        self._image_index = self._load_image_set_index()
        
        # Default to roidb handler
        #govind: So this handler must be overwritten in faster-rcnn since we don't use selective search there?
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'Caltech devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i): # i is a number
        """
        Return the absolute path to image i in the image sequence.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  self._image_index[i] + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    #govind: This function create an image_index object which has list of 
    #all images for that particular _image_set
    #govind: This returns a a list of all image identifiers.
    # e.g. for image 1022.jpg in set01, V003 it stores "set01/V003/1022"
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        #govind: Returns a list of all image names
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'caltech')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # This is all the ground-truth annotations that are extracted from VOC/Annotations directory
        # one-time and then stored. This is done to skip processing of VOC/Annotations
        # for future calls (as the function header says)
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # govind: Forcefully re-read annotations as of now
        if 0:#os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        else:
            imagesetfile = os.path.join(self._data_path, 'ImageSets',
                                          self._image_set + '.txt')
                                          
            # read list of images
            with open(imagesetfile, 'r') as f:
                lines = f.readlines()
            imagenames = [x.strip() for x in lines]                                      
                                          
            caltech_parsed_data = parse_caltech_annotations(imagenames,
                os.path.join(self._data_path, 'annotations'))              
                
            #govind: this is reading the annotations from VOC/Annotations 
            # directory and writing them in data/cache directory.
            # gt_roidb is a list of dictionaries. Nth element of this list 
            # is a corresponding dictionary for Nth image (which is usally
            # different from N.jpg)
            gt_roidb = [self._load_caltech_annotation(caltech_parsed_data, i)
                        for i in self.image_index]
                    
            with open(cache_file, 'wb') as fid:
                cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote gt roidb to {}'.format(cache_file)
            return gt_roidb

    #govind: This alt-opt training log doesn't print the 
    # 'wrote ss roidb to' line. Hence this function is never getting called.
    def selective_search_roidb(self):
        assert 0 #govind: this function might need be modified for Caltech
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    #govind: This guy is somehow combining the gt_roidb and rpn_roidb
    # Don't know why
    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            #govind: For testing, 
            roidb = self._load_rpn_roidb(None)

        return roidb

    #govind: The intermediate proposals during multi-stage training is stored in 
    # output/faster_rcnn_alt_opt/voc_2007_trainval directory. 
    # I believe that this function is used to load those proposals.
    # The config['rpn_file'] is set to None at the start of this file. So 
    # somebody must be updating it during training
    # There are 2 such .pkl (pickel) files:
    #   vgg16_rpn_stage1_iter_100_proposals.pkl
    #   vgg16_rpn_stage2_iter_100_proposals.pkl
    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    #govind: I havent verified it, but I believe this function might not
    # be getting called since it involves selective_search
    def _load_selective_search_roidb(self, gt_roidb):
        assert 0 #govind: this function might need be modified to Caltech
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    #govind: The VOC2007/Annotations directory contains annotations for all
    # the samples. 
    # govind: the parameter <index> is the Image file name (excluding extension and path)
    # govind: Function returns a dictionary corresponding to Annotations of input <image>
    # idx_image is an image name
    def _load_caltech_annotation(self, caltech_parsed_data, idx_image):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        # objs is a list of dictionaries. Each dictionary represents
        # and object
        objs = caltech_parsed_data[idx_image]
        #govind: num_objs is the number of objects in the current image
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        #govind: obj is a dictionary. Containing 'lbl' and 'pol'
        # keys. 
        for ix, obj in enumerate(objs):
            x1 = obj['bbox'][0]
            y1 = obj['bbox'][1]
            x2 = obj['bbox'][2]
            y2 = obj['bbox'][3]
            cls = self._class_to_ind[obj['name']]
            boxes[ix, :] = obj['bbox']
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            
        #govind: Converting the overlap matrix to CSR format
        overlaps = scipy.sparse.csr_matrix(overlaps)
        
        #govind: So the <roidb> is dictionary.
        #I believe that we're not using the values seg_areas pesent in this dictionary
        #maybe legacy-code
        # gt_classes is an array of size <num_objs in the image>
        # contains name of 
        return {'boxes' : boxes,
                'gt_classes': gt_classes, #gt_classes is class index (integer). _background_ is 0
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    #govind: What is _comp_id
    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    #govind: The results of testing:
    #The file is stored as VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_<class_name>.txt
    def _get_caltech_results_file_template(self):
        results_dir = path = os.path.join(
            self._devkit_path, 'results');
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        # caltech/results/<_get_comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        return os.path.join(results_dir, filename)

    #govind:     
    # Every class has a file associated with it. The cotents are
    # <image number> <score?> <4 coordinates> e.g.
    # 000006 0.056 1.0 2.5 388.1 237.0
    # 000045 0.069 366.1 31.8 473.3 267.2
    # 000045 0.056 82.9 71.5 500.0 371.6
    # 000070 0.052 162.7 40.5 323.1 373.2
    def _write_caltech_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} caltech results file'.format(cls)
            filename = self._get_caltech_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    #govind: This function is responsible for evaluating the 
    # performance of network by comparing the 
    # It is executed at the end of testing and is called by 
    # lib/fast_rcnn/test_net()
    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(self._data_path, 'annotations')
        imagesetfile = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')            
            
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        #govind: Write Precision, Recall results of each class 
        # into a separate .pkl file
        for i, cls in enumerate(self._classes):
            #govind: Ignore all other classes, including '__background__'
            if cls != 'person':
                continue
            filename = self._get_caltech_results_file_template().format(cls)
            #govind: It's calling a function which will give Precision, Recall and 
            # average precision if we pass it the _results_file
            rec, prec, ap = caltech_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5, use_07_metric=True)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        #govind: Computing Mean average precision
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    
    def _do_matlab_eval(self, output_dir='output'):
        assert False #govind: This will not be called
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_caltech_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            assert(0) #govind: code not modified for caltech
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            #govind: Remove the temp result files 
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_caltech_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    #govind: know when this part is getting executed
    assert 0

    from datasets.pascal_voc import pascal_voc
    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
