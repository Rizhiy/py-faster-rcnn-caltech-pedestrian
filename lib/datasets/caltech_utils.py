#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
from scipy.io import loadmat
from collections import defaultdict


import xml.etree.ElementTree as ET
import cPickle
import numpy as np

def parse_caltech_annotations(imagenames, ann_dir):
    #recs is a dictionary with keys as imagename.
    #value is a list of dictionaries where each dictionary belongs
    #to an object
    #Inside each dictionary the keys are 'name', 'bbox' etc
    recs = {}
    all_obj = 0
    data = defaultdict(dict)
    
    for dname in sorted(glob.glob(ann_dir+'/set*')):
        set_name = os.path.basename(dname)
        data[set_name] = defaultdict(dict)
        for anno_fn in sorted(glob.glob('{}/*.vbb'.format(dname))):
            vbb = loadmat(anno_fn)
            nFrame = int(vbb['A'][0][0][0][0][0])
            objLists = vbb['A'][0][0][1][0]
            #govind: Is it maximum number of objects in that vbb file?
            maxObj = int(vbb['A'][0][0][2][0][0])
            objInit = vbb['A'][0][0][3][0]
            objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
            objStr = vbb['A'][0][0][5][0]
            objEnd = vbb['A'][0][0][6][0]
            objHide = vbb['A'][0][0][7][0]
            altered = int(vbb['A'][0][0][8][0][0])
            log = vbb['A'][0][0][9][0]
            logLen = int(vbb['A'][0][0][10][0][0])

            video_name = os.path.splitext(os.path.basename(anno_fn))[0]
            #govind: One iteration of this loop processes one frame
            for frame_id, obj in enumerate(objLists):
                objs = []
                if len(obj) > 0:
                    for id, pos in zip(obj['id'][0], obj['pos'][0]):
                        id = int(id[0][0]) - 1  # MATLAB is 1-origin
                        keys = obj.dtype.names
                        pos = pos[0].tolist()
                        datum = dict(zip(keys, [id, pos]))
                        obj_datum = dict()
                        obj_datum['name'] = str(objLbl[datum['id']])
                        #govind: Ignore 'people', 'person?' and 'person-fa' labels
                        if obj_datum['name'] != 'person':
                            continue
                        obj_datum['pose'] = 'Unspecified'
                        obj_datum['truncated'] = 0
                        obj_datum['difficult'] = 0
                        obj_datum['bbox'] = [pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]
                        objs.append(obj_datum)
                data[set_name][video_name][frame_id] = objs
    for imagename in imagenames:
        image_set_name = imagename[0:5]
        image_vid_name = imagename[5:9]
        image_id       = int(imagename[10:])
        if image_id in data[image_set_name][image_vid_name]:
            recs[imagename] = data[image_set_name][image_vid_name][image_id]
        else:
            print "Warning: No %s.jpg found in Annotations" %(imagename)
    return recs

def caltech_ap(rec, prec, use_07_metric=False):
    """ ap = caltech_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def caltech_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    #govind: If testing is performed after training, then 
    # the ground-truth annots would already be present
    # as cachefile
    #govind: unconditionally parse annotations
    if 1:#not os.path.isfile(cachefile):
        # load annots
        #govind: recs is a dictionary with <imagename> as keys
        recs = parse_caltech_annotations(imagenames, annopath)
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    #govind: recs is not class specific. Hence create another 
    # dictionary class_recs which is specific to this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
        #There might not be any objects in the picture
        #if not recs[imagename]: #Check if list is empty
        #    print 'Warn: No labels present for: ', imagename

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    
    #govind: TODO: lines may be empty 
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines] #Image name
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        #load all ground truths for this image
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = caltech_ap(rec, prec, use_07_metric)

    return rec, prec, ap
    