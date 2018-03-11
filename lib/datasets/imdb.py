# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Fuwen Tan @ U.Va (2017)
# --------------------------------------------------------

import os, sys, math
import cv2, cPickle, copy
import os.path as osp
import numpy as np
import scipy.sparse
from config import cfg
import matplotlib.pyplot as plt
import datasets.ds_utils as ds_utils

class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._roidb = None
        self._objdb = None
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, val):
        self._classes = val

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb(self):
        return self._roidb

    @property
    def objdb(self):
        return self._objdb

    @roidb.setter
    def roidb(self, val):
        self._roidb = val

    @objdb.setter
    def objdb(self, val):
        self._objdb = val

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
      return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def append_flipped_images(self):
        num_samples = len(self.roidb)
        for i in range(num_samples):
            entry = copy.deepcopy(self.roidb[i])
            boxes = entry['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = entry['width'] - oldx2 - 1
            boxes[:, 2] = entry['width'] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry['boxes'] = boxes
            entry['flipped'] = True

            # if entry.get('all_boxes', None):
            #     all_boxes = entry['all_boxes'].copy()
            #     oldx1 = all_boxes[:, 0].copy()
            #     oldx2 = all_boxes[:, 2].copy()
            #     all_boxes[:, 0] = entry['width'] - oldx2 - 1
            #     all_boxes[:, 2] = entry['width'] - oldx1 - 1
            #     assert (all_boxes[:, 2] >= all_boxes[:, 0]).all()
            #     entry['all_boxes'] = all_boxes
            # all_boxes = entry['all_boxes'].copy()
            # oldx1 = all_boxes[:, 0].copy()
            # oldx2 = all_boxes[:, 2].copy()
            # all_boxes[:, 0] = entry['width'] - oldx2 - 1
            # all_boxes[:, 2] = entry['width'] - oldx1 - 1
            # assert (all_boxes[:, 2] >= all_boxes[:, 0]).all()
            # entry['all_boxes'] = all_boxes

            self.roidb.append(entry)
        #self._image_index = self._image_index * 2

    def permute_roidb_indices(self):
        self.roidb_perm = np.random.permutation(range(len(self.roidb)))
        self.roidb_cur = 0

    def permute_objdb_indices(self):
        self.objdb_perm = np.random.permutation(range(len(self.objdb)))
        self.objdb_cur = 0

    def get_max_sequence_length(self):
        self.maxSequenceLength = np.amax(np.array([r['boxes'].shape[0] for r in self.roidb]))
        return self.maxSequenceLength

    ####################################################################
    # Visualization
    def draw_roidb_bboxes(self, output_dir, roidb=None):
        ds_utils.maybe_create(output_dir)
        ds_utils.maybe_create(osp.join(output_dir, 'roidb_boxes'))

        if roidb is None:
            roidb = self._roidb

        for i in xrange(len(roidb)):
            roi = roidb[i]
            im_path = roi['image']
            bboxes  = roi['boxes'].copy()
            clses   = roi['clses']

            # image data, flip if necessary
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            if roi['flipped']:
                # print('flipped %d'%i)
                img = cv2.flip(img, 1)

            img, offset_x, offset_y = \
                ds_utils.create_squared_image(img, cfg.PIXEL_MEANS)
            bboxes[:, 0] += offset_x; bboxes[:, 1] += offset_y
            bboxes[:, 2] += offset_x; bboxes[:, 3] += offset_y

            fontScale = 0.0007 * math.sqrt(2 * img.shape[0] * img.shape[0])

            for j in xrange(bboxes.shape[0]):
                bb  = bboxes[j, :].astype(np.int16)
                cls = self.classes[clses[j]]

                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]),
                            (0, 255, 0), 1)

                cv2.putText(img, '{:}_{:}'.format(j, cls),
                            (bb[0], bb[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 255), 1)


            output_path = osp.join(output_dir, 'roidb_bboxes', osp.basename(im_path))
            cv2.imwrite(output_path, img)
            print i

    def draw_objdb_bboxes(self, output_dir, objdb=None):
        ds_utils.maybe_create(output_dir)
        ds_utils.maybe_create(osp.join(output_dir, 'objdb_boxes'))

        if objdb is None:
            objdb = self._objdb

        for i in xrange(len(objdb)):
            obj = objdb[i]
            im_path = obj['image']
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            box = obj['box']
            cls = obj['cls']
            aid = obj['obj_id']

            if obj['flipped']:
                # print('flipped %d'%i)
                img = cv2.flip(img, 1)

            img, offset_x, offset_y = \
                ds_utils.create_squared_image(img, cfg.PIXEL_MEANS)

            box[0] += offset_x; box[1] += offset_y
            box[2] += offset_x; box[3] += offset_y

            bb = box.astype(np.int)

            fontScale = 0.0007 * math.sqrt(2 * img.shape[0] * img.shape[0])

            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]),
                        (0, 255, 0), 1)

            cv2.putText(img, '{:}_{:}'.format(j, cls),
                        (bb[0], bb[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 255), 1)

            im_name, im_ext = osp.splitext(osp.basename(im_path))
            output_path = osp.join(output_dir, 'objdb_boxes', im_name+'_'+str(aid).zfill(12)+im_ext)
            cv2.imwrite(output_path, img)

            print i

    def draw_images(self, output_dir, roidb=None):
        ds_utils.maybe_create(output_dir)
        output_dir = osp.join(output_dir, 'images')
        ds_utils.maybe_create(output_dir)

        if roidb is None:
            roidb = self._roidb

        for i in xrange(len(roidb)):
            roi = roidb[i]
            im_path = roi['image']

            # image data, flip if necessary
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            # img, offset_x, offset_y = \
            #         ds_utils.create_squared_image(img, cfg.PIXEL_MEANS)
            # img = cv2.resize(img, (512, 512))
            output_path = osp.join(output_dir, osp.basename(im_path))
            cv2.imwrite(output_path, img)
            print i, osp.basename(im_path)

    ####################################################################
    # Minibatch

    def get_rnn_minibatch(self, max_seq_len, square=True, vis=False):

        #######################################################################
        # rename the config parameters to make the codes look clear
        batch_size = cfg.TRAIN.BATCH_SIZE
        resolution = cfg.RESOLUTION
        grid_shape = cfg.GRID_SHAPE

        #######################################################################
        # indices of the minibatch
        if self.roidb_cur + batch_size >= len(self.roidb):
            self.permute_roidb_indices()
        db_inds = self.roidb_perm[self.roidb_cur : self.roidb_cur + batch_size]
        self.roidb_cur += batch_size
        #######################################################################

        #######################################################################
        # to be returned
        objects = []; centers = []; ratios  = []; masks   = []
        # normalized xywh representation
        bboxes  = np.zeros((batch_size, max_seq_len, 4), dtype=np.float32)
        # grid box offset
        deltas  = np.zeros((batch_size, max_seq_len, 4), dtype=np.float32)
        images  = np.zeros((batch_size, resolution[0], \
                            resolution[1], resolution[2]), dtype=np.float32)
        #######################################################################

        for i in xrange(batch_size):
            rois     = self.roidb[db_inds[i]]
            im_path  = rois['image']
            width    = rois['width']
            height   = rois['height']
            gt_boxes = rois['boxes'].copy()
            gt_cats  = rois['clses'].copy()
            areas    = rois['seg_areas']

            # number of instances should not exceed max_seq_len
            num_instances = min(gt_boxes.shape[0], max_seq_len)

            # image data, flip if necessary
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            if rois['flipped']:
                # print('flipped %d'%i)
                img = cv2.flip(img, 1)

            # sort the objects in the sequence based on their areas
            order    = np.argsort(areas)[::-1]
            gt_boxes = gt_boxes[order, :]
            gt_cats  = gt_cats[order]
            areas    = areas[order]
            # print areas

            # [x1, y1, x2, y2] to [x, y, w, h]
            gt_boxes = ds_utils.xyxy_to_xywh(gt_boxes)

            # if we need square images
            if square:
                img, offset_x, offset_y = \
                    ds_utils.create_squared_image(img, cfg.PIXEL_MEANS)
                gt_boxes[:,0] += offset_x
                gt_boxes[:,1] += offset_y
                width = height = img.shape[0]

            # normalize
            gt_boxes = ds_utils.normalize_xywh(gt_boxes, width, height)

            # truncate the sequences
            gt_boxes = gt_boxes[:num_instances, :]

            # discreted output positions
            grid_indices = ds_utils.xywh_to_index(gt_boxes, \
                grid_shape[1], grid_shape[0])

            # deltas between grid boxes and ground truth boxes
            grid_boxes  = ds_utils.index_to_xywh(grid_indices, \
                grid_shape[1], grid_shape[0])
            grid_deltas = ds_utils.bbox_transform(grid_boxes, gt_boxes)

            # images of the same shape
            images[i, :, :, :] = cv2.resize(img, (resolution[1], resolution[0]))
            # use the last 'num_instances' objects
            bboxes[i, :num_instances, :] = np.expand_dims(gt_boxes, axis=0)
            # grid offsets
            deltas[i, :num_instances, :] = np.expand_dims(grid_deltas, axis=0)
            # object indicators
            objects.append(gt_cats[:num_instances].tolist())
            # masks for loss function
            masks.append(np.ones((num_instances, )).tolist())
            # grid centers and sizes
            centers.append(grid_indices[:, 0].tolist())
            ratios.append(grid_indices[:, 1].tolist())

        # padding
        objects = pad_sequences(objects, maxlen=max_seq_len,
                      padding='post', truncating='post', value=0.)
        centers = pad_sequences(centers, maxlen=max_seq_len,
                      padding='post', truncating='post', value=0.)
        ratios  = pad_sequences(ratios, maxlen=max_seq_len,
                      padding='post', truncating='post', value=0.)
        masks   = pad_sequences(masks, maxlen=max_seq_len,
                      padding='post', truncating='post', value=0.)

        if vis:
            output_dir = osp.abspath(osp.join(cfg.ROOT_DIR, 'output', \
                                              cfg.EXP_DIR, self.name, \
                                              'rnn_minibatch'))
            if not osp.exists(output_dir):
                os.makedirs(output_dir)

            for i in xrange(batch_size):
                rois = self.roidb[db_inds[i]]
                im_name, im_ext = osp.splitext(osp.basename(rois['image']))
                msk = masks[i, :]

                # ground truth boxes
                ibb = bboxes[i, :, :].copy()
                iid = objects[i, :].copy()
                iim = images[i, :, :, :].copy()

                # grid bboxes
                grid_indices = np.vstack((centers[i,:], ratios[i,:])).transpose()
                gbb = ds_utils.index_to_xywh(grid_indices, grid_shape[1], grid_shape[0])

                # regressed bboxes
                rbb = ds_utils.bbox_transform_inv(gbb, deltas[i,:,:])

                # Denormalize
                ibb = ds_utils.denormalize_xywh(ibb, resolution[1], resolution[0])
                gbb = ds_utils.denormalize_xywh(gbb, resolution[1], resolution[0])
                rbb = ds_utils.denormalize_xywh(rbb, resolution[1], resolution[0])

                ibb = ds_utils.xywh_to_xyxy(ibb, resolution[1], resolution[0])
                gbb = ds_utils.xywh_to_xyxy(gbb, resolution[1], resolution[0])
                rbb = ds_utils.xywh_to_xyxy(rbb, resolution[1], resolution[0])

                # fontScale = 0.0007 * math.sqrt(float(\
                #     resolution[0]*resolution[0]+resolution[1]*resolution[1]))

                for j in xrange(ibb.shape[0]):
                    if msk[j] == 0:
                        break

                    id = iid[j]
                    cls = self.classes[id]

                    # ground truth boxes
                    bb = ibb[j, :].astype(np.int16)
                    cv2.rectangle(iim, (bb[0], bb[1]), (bb[2], bb[3]), \
                                (0, 255, 0), 2)
                    # grid boxes
                    bb = gbb[j, :].astype(np.int16)
                    cv2.rectangle(iim, (bb[0], bb[1]), (bb[2], bb[3]), \
                                (255, 0, 0), 1)
                    # regressed boxes
                    bb = rbb[j, :].astype(np.int16)
                    cv2.rectangle(iim, (bb[0], bb[1]), (bb[2], bb[3]), \
                                (0, 0, 255), 1)
                    # cv2.putText(iim, '{:}_{:}'.format(j, cls), \
                    #             (bb[0], bb[1] - 2), \
                    #             cv2.FONT_HERSHEY_SIMPLEX, \
                    #             fontScale, (0, 0, 255), 1)

                output_path = osp.join(output_dir, '%06d_'%i+im_name+'.jpg')
                cv2.imwrite(output_path, iim)

        return images, objects, bboxes, deltas, centers, ratios, masks

    def get_obj_minibatch(self):
        #######################################################################
        # rename the config parameters to make the codes look clear
        batch_size = cfg.TRAIN.BATCH_SIZE

        #######################################################################
        # indices of the minibatch
        if self.objdb_cur + batch_size >= len(self.objdb):
            self.permute_objdb_indices()
        db_inds = self.objdb_perm[self.objdb_cur : self.objdb_cur + batch_size]
        self.objdb_cur += batch_size
        #######################################################################

        minibatch = [self.objdb[x] for x in db_inds]
        return minibatch

    def get_minibatch(self, square=True):
        # outputs: resized images, normalized xywhs, grids

        batch_size = cfg.TRAIN.BATCH_SIZE
        grid_shape = cfg.GRID_SHAPE
        resolution = cfg.RESOLUTION

        #######################################################################
        # indices of the minibatch
        if self.objdb_cur + batch_size >= len(self.objdb):
            self.permute_objdb_indices()
        db_inds = self.objdb_perm[self.objdb_cur : self.objdb_cur + batch_size]
        self.objdb_cur += batch_size
        #######################################################################

        images  = np.zeros((batch_size, resolution[0], \
                        resolution[1], resolution[2]), dtype=np.float32)
        grids = np.zeros((batch_size, 2))
        boxes = np.zeros((batch_size, 4))

        for i in range(batch_size):
            obj = self.objdb[db_inds[i]]
            im_path = obj['background']
            width   = obj['width']
            height  = obj['height']
            box     = obj['box'].copy()

            # image data, flip if necessary
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            if obj['flipped']:
                # print('flipped %d'%i)
                img = cv2.flip(img, 1)
            xywh = ds_utils.xyxy_to_xywh(box.reshape((1,4))).squeeze()

            # if we need square images
            if square:
                img, offset_x, offset_y = \
                    ds_utils.create_squared_image(img, cfg.PIXEL_MEANS)
                xywh[0] += offset_x
                xywh[1] += offset_y
                width = height = img.shape[0]

            nxywh = ds_utils.normalize_xywh(xywh.reshape((1,4)), width, height).squeeze()
            # discreted output positions
            grid  = ds_utils.boxes_to_indices(nxywh.reshape((1,4)), grid_shape).squeeze()
            # images of the same shape
            images[i, :, :, :] = cv2.resize(img, (resolution[1], resolution[0]))
            grids[i, :] = grid
            boxes[i, :] = nxywh

        return images, boxes, grids

    def get_background_minibatch(self, square=True):
        # outputs: resized images, layouts, normalized xywhs, grids

        batch_size = cfg.TRAIN.BATCH_SIZE
        resolution = cfg.PREDICT_RESOLUTION
        grid_shape = cfg.GRID_SHAPE

        #######################################################################
        # indices of the minibatch
        if self.objdb_cur + batch_size >= len(self.objdb):
            self.permute_objdb_indices()
        db_inds = self.objdb_perm[self.objdb_cur : self.objdb_cur + batch_size]
        self.objdb_cur += batch_size
        #######################################################################

        images  = np.zeros((batch_size, resolution[0], \
            resolution[1], resolution[2]), dtype=np.float32)
        layouts = np.zeros((batch_size, resolution[0], \
            resolution[1], resolution[2]), dtype=np.float32)

        grids = np.zeros((batch_size, 2))
        boxes = np.zeros((batch_size, 4))

        for i in range(batch_size):
            obj = self.objdb[db_inds[i]]

            im_path = obj['bg_image']
            lo_path = obj['bg_layout']

            width   = obj['width']
            height  = obj['height']
            box     = obj['box'].copy()

            # image data, flip if necessary
            im = cv2.imread(im_path, cv2.IMREAD_COLOR)
            lo = cv2.imread(lo_path, cv2.IMREAD_COLOR)

            if obj['flipped']:
                # print('flipped %d'%i)
                im = cv2.flip(im, 1)
                lo = cv2.flip(lo, 1)

            xywh = ds_utils.xyxy_to_xywh(box.reshape((1,4))).squeeze()

            # if we need square images
            if square:
                im, ox, oy = \
                    ds_utils.create_squared_image(im, cfg.PIXEL_MEANS)
                xywh[0] += ox
                xywh[1] += oy
                width = height = im.shape[0]
                lo, ox, oy = \
                    ds_utils.create_squared_image(lo, cfg.PIXEL_MEANS)

            nxywh = ds_utils.normalize_xywh(xywh.reshape((1,4)), width, height).squeeze()
            # discreted output positions
            grid  = ds_utils.boxes_to_indices(nxywh.reshape((1,4)), grid_shape).squeeze()

            # images of the same shape
            images[i]   = cv2.resize(im, (resolution[1], resolution[0]))
            layouts[i]  = cv2.resize(lo, (resolution[1], resolution[0]))

            grids[i, :] = grid
            boxes[i, :] = nxywh

            # print im_path, grid

        return images, layouts, boxes, grids

    def get_scene_minibatch(self, square=True):
        # outputs: resized images, layouts, segmentations, normalized xywhs, grids

        batch_size = cfg.TRAIN.BATCH_SIZE
        resolution = cfg.PREDICT_RESOLUTION
        grid_shape = cfg.GRID_SHAPE
        num_clses  = self.num_classes-1

        #######################################################################
        # indices of the minibatch
        if self.objdb_cur + batch_size >= len(self.objdb):
            self.permute_objdb_indices()
        db_inds = self.objdb_perm[self.objdb_cur : self.objdb_cur + batch_size]
        self.objdb_cur += batch_size
        #######################################################################

        images = np.zeros((batch_size, resolution[0], \
                        resolution[1], resolution[2]), dtype=np.float32)
        scenes = np.zeros((batch_size, resolution[0], \
                        resolution[1], num_clses), dtype=np.float32)
        segs   = np.zeros((batch_size, resolution[0], \
                        resolution[1], resolution[2]), dtype=np.float32)

        grids = np.zeros((batch_size, 2))
        boxes = np.zeros((batch_size, 4))

        for i in range(batch_size):
            obj = self.objdb[db_inds[i]]

            im_path  = obj['background']
            seg_path = obj['out_seg']

            width   = obj['width']
            height  = obj['height']
            box     = obj['box'].copy()

            all_boxes = obj['all_boxes'].copy().reshape((-1,4)).astype(np.int)
            all_clses = obj['all_clses'].copy().flatten()

            # image data, flip if necessary
            img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            seg = cv2.imread(seg_path, cv2.IMREAD_COLOR)

            if obj['flipped']:
                # print('flipped %d'%i)
                img = cv2.flip(img, 1)
                seg = cv2.flip(seg, 1)

            xywh = ds_utils.xyxy_to_xywh(box.reshape((1,4))).squeeze()
            ex_box = box.copy().flatten().astype(np.int)

            # if we need square images
            if square:
                img, offset_x, offset_y = \
                    ds_utils.create_squared_image(img, cfg.PIXEL_MEANS)
                xywh[0] += offset_x
                xywh[1] += offset_y


                ex_box[0] += offset_x
                ex_box[1] += offset_y
                ex_box[2] += offset_x
                ex_box[3] += offset_y

                all_boxes[:, 0] += offset_x
                all_boxes[:, 1] += offset_y
                all_boxes[:, 2] += offset_x
                all_boxes[:, 3] += offset_y
                width = height = img.shape[0]
                seg, offset_x, offset_y = \
                    ds_utils.create_squared_image(seg, cfg.PIXEL_MEANS)

            nxywh = ds_utils.normalize_xywh(xywh.reshape((1,4)), width, height).squeeze()
            # discreted output positions
            grid  = ds_utils.boxes_to_indices(nxywh.reshape((1,4)), grid_shape).squeeze()

            # images of the same shape
            images[i] = cv2.resize(img, (resolution[1], resolution[0]))
            segs[i]   = cv2.resize(seg, (resolution[1], resolution[0]))

            factor    = float(resolution[0])/width
            all_boxes = (factor * all_boxes).astype(np.int)
            ex_box    = (factor * ex_box).astype(np.int)
            scenes[i] = ds_utils.create_scenes(resolution[1], resolution[0], all_boxes, all_clses, ex_box=ex_box, n_cls=num_clses)

            grids[i, :] = grid
            boxes[i, :] = nxywh

        return images, scenes, segs, boxes, grids


    ####################################################################
    # Statistic
    def draw_binary_correlation_stat_graph(self, output_dir, roidb=None):
        # Create the output directory if necessary
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        if not osp.exists(osp.join(output_dir, 'images')):
            os.makedirs(osp.join(output_dir, 'images'))

        # Cache files
        present_cache_file      = osp.join(self.cache_path, self.name + '_present_stats.pkl')
        correlation_cache_file  = osp.join(self.cache_path, self.name + '_correlation_stats.pkl')

        # Load cache files if they exist
        if osp.exists(present_cache_file) and osp.exists(correlation_cache_file):
            with open(present_cache_file, 'rb') as fid:
                present_stats = cPickle.load(fid)
            print '{} present stats loaded from {}'.format(self.name, present_cache_file)

            with open(correlation_cache_file, 'rb') as fid:
                correlation_stats = cPickle.load(fid)
            print '{} correlation stats loaded from {}'.format(self.name, correlation_cache_file)
        # Otherwise, create them
        else:
            if roidb == None:
                roidb = self.roidb
            num_rois = len(roidb)

            # present_stats: the number of pairs
            present_stats     = np.zeros((self.num_classes, self.num_classes))
            correlation_stats = [[ np.zeros((6, 0)) for j in xrange(self.num_classes) ] \
                                                    for i in xrange(self.num_classes) ]

            for i in xrange(num_rois):
                rois      = roidb[i]
                im_width  = float(rois['width'])
                im_height = float(rois['height'])
                bboxes    = rois['boxes'].copy()
                classes   = rois['clses']

                # At least 2 objects
                if bboxes.shape[0] < 2:
                    continue

                # Assume squared images
                max_dim = np.maximum(im_width, im_height)
                nfactor = np.array([max_dim, max_dim, \
                                    max_dim, max_dim]).reshape((1,4))

                # Change representations from xyxy to xywh
                bboxes  = ds_utils.xyxy_to_xywh(bboxes)
                # Normalize
                bboxes  = np.divide(bboxes, nfactor)
                # Area
                areas   = np.multiply(bboxes[:, 2], bboxes[:, 3]).squeeze()
                # Aspect ratio
                ratios  = np.divide(bboxes[:, 2], bboxes[:, 3]).squeeze()

                for j in xrange(bboxes.shape[0] - 1):
                    cls1   = classes[j]
                    bbox1  = bboxes[j, :].squeeze()

                    for k in xrange(j + 1, bboxes.shape[0]):
                        cls2   = classes[k]
                        bbox2  = bboxes[k, :].squeeze()

                        offset = bbox2[:2] - bbox1[:2]

                        correlation21 = np.array([offset[0], offset[1],
                                                  areas[j], areas[k],
                                                  ratios[j], ratios[k]]).reshape((6,1))

                        correlation12 = np.array([-offset[0], -offset[1],
                                                  areas[k],  areas[j],
                                                  ratios[k], ratios[j]]).reshape((6,1))

                        correlation_stats[cls1][cls2] = \
                                np.hstack((correlation_stats[cls1][cls2], correlation21))
                        correlation_stats[cls2][cls1] = \
                                np.hstack((correlation_stats[cls2][cls1], correlation12))

                        present_stats[cls1, cls2] += 1
                        present_stats[cls2, cls1] += 1

                print i


            with open(present_cache_file, 'wb') as fid:
                cPickle.dump(present_stats, fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote present stats to {}'.format(present_cache_file)

            with open(correlation_cache_file, 'wb') as fid:
                cPickle.dump(correlation_stats , fid, cPickle.HIGHEST_PROTOCOL)
            print 'wrote correlation stats to {}'.format(correlation_cache_file)

        plt.switch_backend('agg')
        for i in xrange(1, self.num_classes):
            for j in xrange(1, self.num_classes):
                correlation = correlation_stats[i][j]

                fig = plt.figure()

                plt.hist2d(correlation[0, :], correlation[1, :], 20, range=[[-1.0, 1.0], [-1.0, 1.0]])
                plt.colorbar()
                plt.xlim([-1.0, 1.0])
                plt.ylim([-1.0, 1.0])
                plt.title('offset: %s vs %s'%(self.classes[i], self.classes[j]))
                plt.grid(True)

                fig.savefig(os.path.join(output_dir, 'images/offset_%02d_%02d.jpg' % (i, j)), bbox_inches='tight')
                plt.close(fig)

                fig = plt.figure()

                plt.hist2d(correlation[2, :], correlation[3, :], 20, range=[[0, 0.05], [0, 0.05]])
                plt.colorbar()
                plt.xlim([0, 0.05])
                plt.ylim([0, 0.05])
                plt.title('area: %s vs %s'%(self.classes[i], self.classes[j]))
                plt.grid(True)

                fig.savefig(osp.join(output_dir, 'images/area_%02d_%02d.jpg' % (i, j)), bbox_inches='tight')
                plt.close(fig)


                fig = plt.figure()

                plt.hist2d(correlation[4, :], correlation[5, :], 20, range=[[0, 4.0], [0, 4.0]])
                plt.colorbar()
                plt.xlim([0, 4.0])
                plt.ylim([0, 4.0])
                plt.title('aspect ratio: %s vs %s'%(self.classes[i], self.classes[j]))
                plt.grid(True)

                fig.savefig(osp.join(output_dir, 'images/ratio_%02d_%02d.jpg' % (i, j)), bbox_inches='tight')
                plt.close(fig)

                im1 = cv2.resize(cv2.imread(osp.join(output_dir, 'images/offset_%02d_%02d.jpg' % (i, j))), (648, 545))
                im2 = cv2.resize(cv2.imread(osp.join(output_dir, 'images/area_%02d_%02d.jpg' % (i, j))), (648, 545))
                im3 = cv2.resize(cv2.imread(osp.join(output_dir, 'images/ratio_%02d_%02d.jpg' % (i, j))), (648, 545))

                im = np.zeros((545, 648 * 3, 3), dtype=np.int16)
                im[:,      :  648, :] = im1
                im[:,   648:2*648, :] = im2
                im[:, 2*648:3*648, :] = im3

                cv2.imwrite(osp.join(output_dir, 'images/%02d_%02d.jpg' % (i, j)), im)

                print i,j

    def create_binary_correlation_stat_html(self, output_dir, roidb=None):
        from html import HTML
        # Create the directory if necessary
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        present_cache_file = osp.join(self.cache_path, self.name + '_present_stats.pkl')
        assert os.path.exists(present_cache_file)

        with open(present_cache_file, 'rb') as fid:
            present_stats = cPickle.load(fid)
        print '{} present stats loaded from {}'.format(self.name, present_cache_file)

        config_html = HTML()
        config_table = config_html.table(border='1')

        for i in xrange(self.num_classes):
            r = config_table.tr
            if i == 0:
                r.th('---')
            else:
                r.th('%s'%self.classes[i])
            for j in xrange(1, self.num_classes):
                c = r.td
                if i == 0:
                    c.a('%s'%self.classes[j])
                else:
                    c.a('%d'%int(present_stats[i, j]), href='images/%02d_%02d.jpg'%(i,j))

        html_file = open(osp.join(output_dir, 'coco_offsets_table.html'), 'w')
        print >> html_file, config_table
        html_file.close()
