# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Fuwen Tan @ U.Va (2017)
# --------------------------------------------------------

import os, math
import sys, copy
import cv2, json
import numpy as np
import os.path as osp
import cPickle
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from datasets.cython_bbox import bbox_overlaps
from config import cfg
from glob import glob
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from PIL import Image

class coco(imdb):
    def __init__(self, image_set, year):

        imdb.__init__(self, 'coco_' + year + '_' + image_set)

        self._year      = year
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'coco')

        # load COCO API, classes, class <-> id mappings
        self._COCO = COCO(self._get_instances_ann_file())
        self._image_index = self._load_image_set_index()
        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats],
                                              self._COCO.getCatIds()))

        # Some image sets are "views" (i.e. subsets) into others.
        # For example, minival2014 is a random 5000 image subset of val2014.
        # This mapping tells us where the view's images and proposals come from.
        self._view_map = {
            'minival2014' : 'val2014',          # 5k val2014 subset
            'valminusminival2014' : 'val2014',  # val2014 \setminus minival2014
        }
        coco_name = image_set + year  # e.g., "val2014"
        self._data_name = (self._view_map[coco_name]
                           if self._view_map.has_key(coco_name)
                           else coco_name)
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        self._gt_splits = ('train', 'val', 'minival')
        self._roidb = self.gt_roidb()
        self.filter_roidb()

        file_path = osp.join(self._data_path, self._image_set+self._year+'_filtered_persons.txt')
        assert osp.exists(file_path), 'Path does not exist: {}'.format(file_path)
        self._cand_objs = set(np.loadtxt(file_path, dtype=str))
         
        if cfg.TRAIN.USE_FLIPPED:
            print 'Appending horizontally-flipped examples...'
            self.append_flipped_images()
            print 'done'

        self.permute_roidb_indices()
        print('roidb entries: {}'.format(len(self._roidb)))

        self._objdb = self.gt_objdb()
        self.permute_objdb_indices()
        print('objdb entries: {}'.format(len(self._objdb)))

    def _get_instances_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 \
                             else 'image_info'
        return osp.join(self._data_path, 'annotations', \
                        prefix + '_' + self._image_set + self._year + '.json')

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self._COCO.getImgIds()
        return image_ids

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = ('COCO_' + self._data_name + '_' + str(index).zfill(12) + '.jpg')
        image_path = osp.join(self._data_path, 'images', self._data_name, file_name)
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def alpha_path_from_indices(self, img_id, ann_id):
        file_name = ('COCO_' + self._data_name + '_' + str(img_id).zfill(12) + '_' + str(ann_id).zfill(12) + '.jpg')
        file_path = osp.join(self._data_path, 'alphas', 'person', self._data_name, file_name)
        # assert osp.exists(file_path), 'Path does not exist: {}'.format(file_path)
        return file_path

    def inpainted_path_from_indices(self, img_id, ann_id):
        file_name = ('COCO_' + self._data_name + '_' + str(img_id).zfill(12) + '_' + str(ann_id).zfill(12) + '.jpg')
        file_path = osp.join(self._data_path, 'inpainted', 'person', self._data_name, file_name)
        # assert osp.exists(file_path), 'Path does not exist: {}'.format(file_path)
        return file_path

    def crop_feature_path_from_indices(self, img_id, ann_id):
        file_name = ('COCO_' + self._data_name + '_' + str(img_id).zfill(12) + '_' + str(ann_id).zfill(12) + '.npy')
        file_path = osp.join(self._data_path, 'features', 'crop', cfg.LAYER, self._data_name, file_name)
        # assert osp.exists(file_path), 'Path does not exist: {}'.format(file_path)
        return file_path

    def full_feature_path_from_indices(self, img_id, ann_id):
        file_name = ('COCO_' + self._data_name + '_' + str(img_id).zfill(12) + '_' + str(ann_id).zfill(12) + '.npy')
        file_path = osp.join(self._data_path, 'features', 'full', cfg.LAYER, self._data_name, file_name)
        # assert osp.exists(file_path), 'Path does not exist: {}'.format(file_path)
        return file_path

    def gist_feature_path_from_indices(self, img_id, ann_id):
        file_name = ('COCO_' + self._data_name + '_' + str(img_id).zfill(12) + '_' + str(ann_id).zfill(12) + '.npy')
        file_path = osp.join(self._data_path, 'features', 'gist', self._data_name, file_name)
        # assert osp.exists(file_path), 'Path does not exist: {}'.format(file_path)
        return file_path

    def background_layout_path_from_indices(self, img_id, ann_id):
        file_name = ('COCO_' + self._data_name + '_' + str(img_id).zfill(12) + '_' + str(ann_id).zfill(12) + '.jpg')
        file_path = osp.join(self._data_path, 'background_layouts', 'person', self._data_name, file_name)
        # assert osp.exists(file_path), 'Path does not exist: {}'.format(file_path)
        return file_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_coco_annotation(index) for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def gt_objdb(self, roidb=None):
        if roidb == None:
            roidb = self.roidb

        gt_objdb = []
        for x in roidb:
            gt_objdb = gt_objdb + self._roi_to_obj(x)

        return gt_objdb

    def _roi_to_obj(self, roi):
        boxes     = roi['boxes']
        clses     = roi['clses']
        seg_areas = roi['seg_areas']
        polys     = roi['polys']
        annIDs    = roi['ann_ids']
        img_id    = roi['image_index']

        obj = []

        for i in xrange(boxes.shape[0]):
            obj_id = annIDs[i]
            name = ('COCO_' + self._data_name + '_' + str(img_id).zfill(12) + '_' + str(obj_id).zfill(12) + '.jpg')
            if not (name in self._cand_objs):
                 continue
            
            obj.append(
            { 'image' : roi['image'],
              'name' : ('COCO_' + self._data_name + '_' + str(img_id).zfill(12) + '_' + str(obj_id).zfill(12)),
              'alpha' : self.alpha_path_from_indices(img_id, obj_id),
              'bg_image'  : self.inpainted_path_from_indices(img_id, obj_id),
              'bg_layout' : self.background_layout_path_from_indices(img_id, obj_id),
              'crop_feat' : self.crop_feature_path_from_indices(img_id, obj_id),
              'full_feat' : self.full_feature_path_from_indices(img_id, obj_id),
              'gist_feat' : self.gist_feature_path_from_indices(img_id, obj_id),
              'width'  : roi['width'],
              'height' : roi['height'],
              'img_id' : img_id,
              'obj_id' : obj_id,
              'area'   : seg_areas[i],
              'poly'   : polys[i],
              'box'    : boxes[i, :],
              'cls'    : clses[i],
              'flipped' : roi['flipped'],
              'all_boxes' : roi['all_boxes'],
              'all_clses' : roi['all_clses']})

        return obj

    def _load_coco_annotation(self, index):
        """
        Loads COCO bounding-box & segmentation instance annotations.
        Crowd instances are removed.
        """
        im_ann = self._COCO.loadImgs(index)[0]
        width  = im_ann['width']; height = im_ann['height']

        #######################################################################
        # get bboxes that are outside crowd regions
        annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=False)
        #######################################################################

        objs = self._COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_IDs = []
        for i in xrange(len(objs)):
            obj = objs[i]
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1 and (not obj['iscrowd']):
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_IDs.append(annIds[i])

        ########################################################################
        boxes      = []
        gt_classes = []
        seg_areas  = []
        # RLE representation of binary mask
        rles       = []
        #######################################################################

        #######################################################################
        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                          for cls in self._classes[1:]])

        for i in xrange(len(valid_objs)):
            obj = valid_objs[i]
            cls = coco_cat_id_to_class_ind[obj['category_id']]

            #######################################################################
            if type(obj['segmentation']) == list:
                # polygon
                rle = COCOmask.frPyObjects(obj['segmentation'], height, width)
            elif type(obj['segmentation']['counts']) == list:
                rle = COCOmask.frPyObjects([obj['segmentation']], height, width)
            else:
                rle = [obj['segmentation']]
            #######################################################################

            boxes.append(obj['clean_bbox'])
            gt_classes.append(cls)
            seg_areas.append(obj['area'])
            rles.append(rle)

        ###############################################################
        ## calculate the areas of objects
        area = float(width * height)
        mask = np.zeros((height, width), dtype=np.float32)
        for j in xrange(len(rles)):
            rle  = rles[j]
            cur_mask = np.amax(COCOmask.decode(rle), axis=2)
            mask = np.maximum(mask, cur_mask)
        seg_area = np.sum(mask)
        seg_ratio = seg_area/area
        # print seg_ratio
        ###############################################################

        return {'image'  : self.image_path_from_index(index),
                'width'  : width,
                'height' : height,
                'boxes'  : np.array(boxes).reshape((-1,4)),
                'clses'  : np.array(gt_classes),
                'polys'     : rles,
                'ann_ids'   : np.array(valid_IDs),
                'flipped'   : False,
                'seg_areas' : np.array(seg_areas),
                'image_index': index,
                'seg_ratio': seg_ratio}

    ####################################################################
    # Filtering
    ####################################################################

    def select_objects(self, roi, indices):

        if len(indices) == 0:
            roi['boxes'] = np.zeros((0, 4))
            roi['clses'] = np.zeros((0,))
            roi['seg_areas'] = np.zeros((0,))
            roi['polys'] = []
            roi['ann_ids'] = np.zeros((0,))
        else:
            clses = roi['clses']
            areas = roi['seg_areas']
            polys = roi['polys']
            tmps  = roi['boxes']
            annIDs = roi['ann_ids']
            roi['boxes'] = tmps[indices, :].reshape((-1,4))
            roi['clses'] = clses[indices]
            roi['polys'] = [polys[x] for x in indices]
            roi['seg_areas'] = areas[indices]
            roi['ann_ids'] = annIDs[indices]

        return roi

    def filter_small_objects(self, roi):
        areas = roi['seg_areas']
        if len(areas) < 1:
            return roi

        indices = np.where(areas > cfg.TRAIN.MIN_SEG_AREA)[0]
        return self.select_objects(roi, indices)

    def filter_overlap_objects(self, roi):
        boxes = roi['boxes'].copy()
        if boxes.shape[0] < 2:
            return roi

        tmp_boxes = boxes.copy()
        overlaps = bbox_overlaps(boxes.astype(np.float), \
                                 tmp_boxes.astype(np.float))

        for i in xrange(boxes.shape[0]):
            overlaps[i,i] = 0.0

        overlaps = np.amax(overlaps, axis=-1)
        indices  = np.where(overlaps < 0.2)[0]

        return self.select_objects(roi, indices)

    def filter_boundary_objects(self, roi):
        boxes = roi['boxes'].copy()
        if boxes.shape[0] < 1:
            return roi

        width  = roi['width']
        height = roi['height']

        # xywh = ds_utils.xyxy_to_xywh(boxes)
        boxes[:,0] -= cfg.TRAIN.DIST_TO_EDGE #0.1 * xywh[:,2]
        boxes[:,2] += cfg.TRAIN.DIST_TO_EDGE #0.1 * xywh[:,2]
        boxes[:,1] -= cfg.TRAIN.DIST_TO_EDGE #0.1 * xywh[:,3]
        boxes[:,3] += cfg.TRAIN.DIST_TO_EDGE #0.1 * xywh[:,3]

        indices = []
        for i in xrange(boxes.shape[0]):
            if boxes[i, 0] >= 0 and boxes[i, 1] >= 0 and \
               boxes[i, 2] < width and boxes[i, 3] < height:
                indices.append(i)

        return self.select_objects(roi, indices)

    def filter_non_person(self, roi):
        clses = roi['clses'].copy()
        if len(clses) < 1:
            return roi
        # 'person' has ID 1
        indices = np.where(clses == 1)[0]

        return self.select_objects(roi, indices)

    def filter_roidb(self):
        def is_valid(entry):
            valid = entry['boxes'].shape[0] > 0
            return valid

        num = len(self.roidb)
        self.roidb = [self.filter_small_objects(x)    for x in self.roidb]
        # needed to render the layouts
        for x in self.roidb:
            x['all_boxes'] = x['boxes'].copy()
            x['all_clses'] = x['clses'].copy()
        # self.roidb = [self.filter_overlap_objects(x)  for x in self.roidb]
        # self.roidb = [self.filter_boundary_objects(x) for x in self.roidb]
        self.roidb = [self.filter_non_person(x)       for x in self.roidb]
        
        filtered_roidb = [entry for entry in self.roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        print('Filtered {} roidb entries: {} -> {} '.format(num - num_after, num, num_after))
        self.roidb = filtered_roidb

    ####################################################################
    # Feature
    ####################################################################

    def dump_full_features(self, output_dir, image_encoder, ctxdb=None):
        full_resolution = cfg.RETRIEVAL_RESOLUTION

        if ctxdb == None:
            ctxdb = self.objdb

        ds_utils.maybe_create(output_dir)
        output_dir = osp.join(output_dir, 'full')
        ds_utils.maybe_create(output_dir)
        output_dir = osp.join(output_dir, cfg.LAYER)
        ds_utils.maybe_create(output_dir)
        output_dir = osp.join(output_dir, '{}'.format(self._image_set+self._year))
        ds_utils.maybe_create(output_dir)

        for i in xrange(len(ctxdb)):
            ctx     = ctxdb[i]
            im_path = ctx['image']
            ann_id  = ctx['obj_id']
            bb      = ctx['box'].copy().astype(np.int)

            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            # img[bb[1]:(bb[3] + 1), bb[0]:(bb[2] + 1), :] = cfg.PIXEL_MEANS.reshape((1,1,3))
            img = cv2.resize(img, (full_resolution[1], full_resolution[0]))

            x = np.expand_dims(img.astype(np.float64), axis=0) - cfg.PIXEL_MEANS.reshape((1,1,1,3))
            features = image_encoder.predict(x).flatten()

            im_name, im_ext = osp.splitext(osp.basename(im_path))
            output_path = osp.join(output_dir, im_name+'_'+str(ann_id).zfill(12)+'.npy')
            with open(output_path , 'wb') as fid:
                cPickle.dump(features, fid, cPickle.HIGHEST_PROTOCOL)
            print i

    def dump_crop_features(self, output_dir, image_encoder, ctxdb=None):
        full_resolution = [224, 224, 3]
        crop_resolution = [112, 112, 3]

        if ctxdb == None:
            ctxdb = self.objdb

        ds_utils.maybe_create(output_dir)
        output_dir = osp.join(output_dir, 'crop')
        ds_utils.maybe_create(output_dir)
        output_dir = osp.join(output_dir, cfg.LAYER)
        ds_utils.maybe_create(output_dir)
        output_dir = osp.join(output_dir, '{}'.format(self._image_set+self._year))
        ds_utils.maybe_create(output_dir)

        for i in xrange(len(ctxdb)):
            ctx     = ctxdb[i]
            im_path = ctx['image']
            ann_id  = ctx['obj_id']
            bb      = ctx['box'].copy().astype(np.int)

            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            img[bb[1]:(bb[3] + 1), bb[0]:(bb[2] + 1), :] = cfg.PIXEL_MEANS.reshape((1,1,3))
            img = ds_utils.crop_and_resize(img, bb, full_resolution, crop_resolution)

            x = np.expand_dims(img.astype(np.float64), axis=0) - cfg.PIXEL_MEANS.reshape((1,1,1,3))
            features = image_encoder.predict(x).flatten()

            im_name, im_ext = osp.splitext(osp.basename(im_path))
            output_path = osp.join(output_dir, im_name+'_'+str(ann_id).zfill(12)+'.npy')
            with open(output_path , 'wb') as fid:
                cPickle.dump(features, fid, cPickle.HIGHEST_PROTOCOL)
            print i

    def extract_gist_feature(self, img_path):
        import leargist
        resolution = cfg.RETRIEVAL_RESOLUTION
        img = Image.open(img_path)
        img = img.resize((resolution[1], resolution[0]))
        feat = leargist.color_gist(img).flatten()
        return feat

    def dump_gist_features(self, output_dir, ctxdb=None):

        if ctxdb == None:
            ctxdb = self.objdb

        ds_utils.maybe_create(output_dir)
        output_dir = osp.join(output_dir, 'gist')
        ds_utils.maybe_create(output_dir)
        output_dir = osp.join(output_dir, '{}'.format(self._image_set+self._year))
        ds_utils.maybe_create(output_dir)

        for i in xrange(len(ctxdb)):
            ctx     = ctxdb[i]
            # print ctx
            im_path = ctx['image']
            ann_id  = ctx['obj_id']
            gist_feat = self.extract_gist_feature(im_path)

            im_name, im_ext = osp.splitext(osp.basename(im_path))
            output_path = osp.join(output_dir, im_name+'_'+str(ann_id).zfill(12)+'.npy')
            with open(output_path , 'wb') as fid:
                cPickle.dump(gist_feat, fid, cPickle.HIGHEST_PROTOCOL)
            print i

    ####################################################################
    # Visualization
    ####################################################################
    def draw_objdb_masks(self, output_dir, objdb=None):
        if objdb == None:
            objdb = self.objdb

        mask_dir = osp.join(output_dir, '{}_objdb_masks'.format(self._image_set))
        img_dir  = osp.join(output_dir, '{}_objdb_imgs'.format(self._image_set))

        ds_utils.maybe_create(output_dir)
        ds_utils.maybe_create(mask_dir)
        ds_utils.maybe_create(img_dir)


        for i in xrange(len(objdb)):
            obj     = objdb[i]

            im_path = obj['image']
            ann_id  = obj['obj_id']
            poly    = obj['poly']
            bb      = obj['box'].astype(np.int16)
            cls     = obj['cls']
            width   = obj['width']
            height  = obj['height']

            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            msk = np.amax(COCOmask.decode(poly), axis=2)

            # binarize the mask
            msk = msk * 255
            retVal, msk = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)
            msk = msk.astype(np.uint8)
            # msk = ds_utils.dilate_mask(msk, 9)

            # img = (1 - 0.5/255 * msk.reshape((height, width, 1))) * img + \
            #       0.5/255 * msk.reshape((height, width, 1)) * \
            #       np.random.random((1, 3)) * 255

            # cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), \
            #             (0, 255, 0), 2)
            #
            # fontScale = 0.0009 * math.sqrt(float(width*width + height*height))
            #
            #
            # cv2.putText(img, '{:}'.format(self.classes[cls]), \
            #             (bb[0], bb[1] - 2), \
            #             cv2.FONT_HERSHEY_SIMPLEX, \
            #             fontScale, (0, 0, 255), 1)

            im_name, im_ext = osp.splitext(osp.basename(im_path))

            output_path = osp.join(mask_dir, im_name+'_'+str(ann_id).zfill(12)+im_ext)
            # output_path = osp.join(mask_dir, im_name+im_ext)
            cv2.imwrite(output_path, msk)

            output_path = osp.join(img_dir,  im_name+'_'+str(ann_id).zfill(12)+im_ext)
            # output_path = osp.join(img_dir,  im_name+im_ext)
            cv2.imwrite(output_path, img)
            print i

    def render_entry_boxes(self, entry, cur_box, color_palette):
        width  = entry['width']
        height = entry['height']

        colors = np.zeros((height, width, 3), dtype=np.float)
        counts = np.zeros((height, width),    dtype=np.float)
        boxes  = np.array(entry['all_boxes']).reshape((-1,4))
        clses  = np.array(entry['all_clses']).flatten()

        for i in range(boxes.shape[0]):
            bb   = boxes[i].astype(np.int)
            diff = np.sum(np.absolute(bb - cur_box))
            if diff < 4:
                continue

            cur_color = np.zeros((height, width, 3))
            cur_count = np.zeros((height, width))

            bb  = boxes[i].astype(np.int)
            cls = clses[i]
            rgb = np.array(color_palette[cls])
            cur_color[bb[1]:(bb[3]+1), bb[0]:(bb[2]+1), :] = rgb.reshape((1,1,3))
            cur_count[bb[1]:(bb[3]+1), bb[0]:(bb[2]+1)] = 1

            colors = colors + cur_color
            counts = counts + cur_count

        counts = counts + 1e-3


        average = np.divide(colors, np.expand_dims(counts, axis=-1))
        output = average.copy().astype(np.int)
        output[average > 255] = 255
        output[average < 0] = 0

        sum_output = np.sum(output, axis=-1)
        sum_output = sum_output.flatten()
        indices = np.where(sum_output < 0.5)[0]

        output = output.reshape((-1, 3))
        output[indices, :] = np.array(color_palette[0]).flatten()
        output = output.reshape((height, width, 3))

        return output

    def draw_objdb_layouts(self, color_palette, output_dir, objdb=None):
        if objdb == None:
            objdb = self.objdb

        layout_dir = osp.join(output_dir, self._image_set+self._year)
        ds_utils.maybe_create(layout_dir)

        for i in range(len(objdb)):
            entry   = objdb[i]
            cur_box = entry['box']
            ann_id  = entry['obj_id']
            output_img  = self.render_entry_boxes(entry, cur_box, color_palette)

            im_path = entry['image']
            im_name, im_ext = osp.splitext(osp.basename(im_path))
            output_path = osp.join(layout_dir, im_name+'_'+str(ann_id).zfill(12)+im_ext)
            cv2.imwrite(output_path, output_img)
            print i

    def draw_roidb_masks(self, output_dir, roidb=None):

        mask_dir = osp.join(output_dir, '{}_roidb_masks'.format(self._image_set))
        img_dir  = osp.join(output_dir, '{}_roidb_imgs'.format(self._image_set))

        ds_utils.maybe_create(output_dir)
        ds_utils.maybe_create(mask_dir)
        ds_utils.maybe_create(img_dir)

        if roidb == None:
            roidb = self.roidb

        for i in xrange(len(roidb)):
            rois    = roidb[i]
            im_path = rois['image']
            clses   = rois['clses']
            boxes   = rois['boxes']
            rles    = rois['polys']
            width   = rois['width']
            height  = rois['height']

            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            msk = np.zeros((height, width), dtype=np.uint8)

            for j in xrange(len(rles)):
                rle = rles[j]
                bb  = boxes[j,:].astype(np.int)
                cls = clses[j]

                tmp = np.amax(COCOmask.decode(rle), axis=2) * 255
                retVal, tmp = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY)
                tmp = tmp.astype(np.uint8)
                tmp = ds_utils.dilate_mask(tmp, 9)
                msk = np.maximum(msk, tmp)

                # fontScale = 0.0009 * math.sqrt(float(width*width + height*height))
                # cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), \
                #             (0, 255, 0), 2)
                # cv2.putText(img, '{:}'.format(self.classes[cls]), \
                #             (bb[0], bb[1] - 2), \
                #             cv2.FONT_HERSHEY_SIMPLEX, \
                #             fontScale, (0, 0, 255), 1)

            # img = (1 - 0.5/255 * msk.reshape((height, width, 1))) * img + \
            #       0.5/255 * msk.reshape((height, width, 1)) * \
            #       np.random.random((1, 3)) * 255


            output_path = osp.join(mask_dir, osp.basename(im_path))
            cv2.imwrite(output_path, msk)
            output_path = osp.join(img_dir,  osp.basename(im_path))
            cv2.imwrite(output_path, img)

            print i
    
    def render_entry_scenes(self, entry, ann_id):
        width  = entry['width']
        height = entry['height']


        vol = np.zeros((height, width, self.num_classes - 1), dtype=np.float)

        boxes  = np.array(entry['all_boxes']).reshape((-1,4))
        clses  = np.array(entry['all_clses']).flatten()
        annIDs = np.array(entry['all_ids']).flatten()

        for i in range(boxes.shape[0]):
            if annIDs[i] == ann_id:
                continue

            bb = boxes[i].astype(np.int)
            cls = clses[i] - 1

            vol[bb[1]:(bb[3]+1), bb[0]:(bb[2]+1), cls] = 255

        return vol

    def draw_objdb_scenes(self, output_dir, objdb=None):
        if objdb == None:
            objdb = self.objdb

        ds_utils.maybe_create(output_dir)

        for i in range(len(objdb)):
            entry = objdb[i]
            ann_id  = entry['obj_id']
            output_vol = self.render_entry_scenes(entry, ann_id)

            im_path = entry['image']
            im_name, im_ext = osp.splitext(osp.basename(im_path))
            new_name = im_name+'_'+str(ann_id).zfill(12)
            output_path = osp.join(output_dir, new_name + '.pkl')

            with open(output_path, 'wb') as fid:
                cPickle.dump(output_vol, fid, cPickle.HIGHEST_PROTOCOL)
            print i

            output_path = new_name + '.jpg'
            cv2.imwrite(output_path, output_vol[:,:,0].astype(np.uint8))

    def draw_position_histogram(self, ctxdb=None):

        resolution = [15, 15]

        bins = np.zeros((resolution[0], resolution[1]))

        X = []
        Y = []

        if ctxdb == None:
            ctxdb = self.objdb

        num_samples = len(ctxdb)

        for i in range(num_samples):
            entry  = ctxdb[i]
            xyxy   = np.array(entry['box']).copy()
            width  = entry['width']
            height = entry['height']

            max_dim = np.maximum(width, height)
            ox = int(0.5 * (max_dim - width))
            oy = int(0.5 * (max_dim - height))

            xyxy[0] += ox; xyxy[1] += oy;
            xyxy[2] += ox; xyxy[3] += oy;

            xywh = ds_utils.xyxy_to_xywh(xyxy.reshape((1, 4))).flatten()
            xywh /= float(max_dim)
            scaled_xy = np.ceil(xywh[:2] * resolution[0])
            scaled_xy = np.maximum(0, scaled_xy-1).astype(np.int)

            bins[scaled_xy[1], scaled_xy[0]] += 1.0
            X = X + [xywh[0]]
            Y = Y + [1.0 - xywh[1]]

            if i%1000 == 0:
                print i


        plt.switch_backend('agg')
        fig = plt.figure()

        plt.hist2d(X, Y, 15, range=[[0.0, 1.0], [0.0, 1.0]])
        plt.colorbar()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        # plt.title('offset: %s vs %s'%(self.classes[i], self.classes[j]))
        plt.grid(True)

        fig.savefig('gt_pos_hist.jpg', bbox_inches='tight')
        plt.close(fig)

        print 'Done'




            # # validate
            # img = cv2.imread(entry['image'], cv2.IMREAD_COLOR)
            # img, _, _ = ds_utils.create_squared_image(img, cfg.PIXEL_MEANS)
            # bb  = xyxy.astype(np.int)
            # cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
            # cv2.imwrite('%09d.jpg'%i, img)
            # print i

    def draw_size_histogram(self, ctxdb=None):

        if ctxdb == None:
            ctxdb = self.objdb

        A = []
        num_samples = len(ctxdb)

        for i in range(num_samples):
            entry = ctxdb[i]
            xyxy   = np.array(entry['box']).copy()
            width  = entry['width']
            height = entry['height']

            max_dim = np.maximum(width, height)
            ox = int(0.5 * (max_dim - width))
            oy = int(0.5 * (max_dim - height))

            xyxy[0] += ox; xyxy[1] += oy;
            xyxy[2] += ox; xyxy[3] += oy;

            xywh = ds_utils.xyxy_to_xywh(xyxy.reshape((1, 4))).flatten()
            xywh /= float(max_dim)

            area = xywh[2] * xywh[3]
            A = A + [area]

            if i%1000 == 0:
                print i

        plt.switch_backend('agg')
        fig = plt.figure()

        plt.hist(A, 100, range=[0.0, 1.0])
        plt.xlim([0.0, 1.0])

        fig.savefig('gt_area_hist.jpg', bbox_inches='tight')
        plt.close(fig)

        print 'Done'

    def draw_ratio_histogram(self, ctxdb=None):

        if ctxdb == None:
            ctxdb = self.objdb

        R = []
        num_samples = len(ctxdb)

        for i in range(num_samples):
            entry = ctxdb[i]
            xyxy   = np.array(entry['box']).copy()
            width  = entry['width']
            height = entry['height']

            max_dim = np.maximum(width, height)
            ox = int(0.5 * (max_dim - width))
            oy = int(0.5 * (max_dim - height))

            xyxy[0] += ox; xyxy[1] += oy;
            xyxy[2] += ox; xyxy[3] += oy;

            xywh = ds_utils.xyxy_to_xywh(xyxy.reshape((1, 4))).flatten()
            xywh /= float(max_dim)

            ratio = np.log(xywh[3] / xywh[2])
            R = R + [ratio]

            if i%1000 == 0:
                print i

        plt.switch_backend('agg')
        fig = plt.figure()

        plt.hist(R, 100, range=[0.0, 2.0])
        plt.xlim([0.0, 2.0])

        fig.savefig('gt_logratio_hist.jpg', bbox_inches='tight')
        plt.close(fig)

        print 'Done'

    def areas_to_heatmap(self, areas, factor = 1024):

        res = len(areas)

        heatmap = np.zeros((res, res))

        for i in range(res):
            heatmap[i, :] = min(int(factor * areas[i]), 255)

        return heatmap

    def draw_heatmap(self, ctxdb=None):

        if ctxdb == None:
            ctxdb = self.objdb

        scale = 15

        areas = np.zeros((scale,))
        count = cfg.EPS * np.ones((scale,))

        num_samples = len(ctxdb)

        for i in range(num_samples):
            entry = ctxdb[i]
            xyxy   = np.array(entry['box']).copy()
            width  = entry['width']
            height = entry['height']

            max_dim = np.maximum(width, height)
            ox = int(0.5 * (max_dim - width))
            oy = int(0.5 * (max_dim - height))

            xyxy[0] += ox; xyxy[1] += oy;
            xyxy[2] += ox; xyxy[3] += oy;

            xywh = ds_utils.xyxy_to_xywh(xyxy.reshape((1, 4))).flatten()
            xywh /= float(max_dim)

            area = xywh[2] * xywh[3]

            scaled_xy = np.ceil(xywh[:2] * scale)
            scaled_xy = np.maximum(0, scaled_xy-1).astype(np.int)

            areas[scaled_xy[1]] += area
            count[scaled_xy[1]] += 1.0

            if i%1000 == 0:
                print i

        areas = np.divide(areas, count)

        heatmap = self.areas_to_heatmap(areas)

        heatmap = cv2.resize(heatmap, (512, 512))
        cv2.imwrite('heatmap.png', heatmap)

    def draw_evaluation_layouts(self, output_dir, isRandom, ctxdb=None):

        if ctxdb == None:
            ctxdb = self.objdb

        num_samples = len(ctxdb)

        for i in range(num_samples):
            entry = ctxdb[i]
            xyxy  = np.array(entry['box']).copy().astype(np.int)
            lyo_path = entry['layout']
            # im_name, im_ext = osp.splitext(osp.basename(lyo_path))

            layout = cv2.imread(lyo_path, cv2.IMREAD_COLOR)

            if isRandom:
                width  = entry['width']
                height = entry['height']
                max_dim = np.maximum(width, height)
                ox = int(0.5 * (max_dim - width))
                oy = int(0.5 * (max_dim - height))

                cen_id  = np.random.randint(0, 15 * 15)
                size_id = np.random.randint(0, 15 * 15)
                box_id  = np.array([cen_id, size_id]).reshape((1,2))

                xywh = ds_utils.indices_to_boxes(box_id, [15, 15, 15, 15])
                xywh = ds_utils.denormalize_xywh(xywh, max_dim, max_dim)
                xyxy = ds_utils.xywh_to_xyxy(xywh, max_dim, max_dim).flatten().astype(np.int)

                xyxy[0] -= ox; xyxy[1] -= oy;
                xyxy[2] -= ox; xyxy[3] -= oy;




            layout[xyxy[1]:(xyxy[3]+1), xyxy[0]:(xyxy[2]+1), :] = 0
            output_path = osp.join(output_dir, osp.basename(lyo_path))

            cv2.imwrite(output_path, layout)
            print i
