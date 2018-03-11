#!/usr/bin/env python

import os, sys
import cv2, json
import time, math
import copy, cPickle
import numpy as np
import os.path as osp
from config import cfg
import datasets.ds_utils as ds_utils
from datasets.cython_bbox import bbox_overlaps
from scipy.spatial.distance import cosine
from sklearn.neighbors import BallTree
from keras.layers import Input, Flatten
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50

# def cus_distance_hybrid(X, Y, **kwargs):
#     xbb = X[:4]; ybb = Y[:4]
#     if ds_utils.xywh_IoU(xbb, ybb) < cfg.TEST.IOU_THRESHOLD:
#         return np.finfo(np.float32).max
#     crop_dist = cosine(X[4:(cfg.FEAT_DIMS[-1]+4)], Y[4:(cfg.FEAT_DIMS[-1]+4)])
#     full_dist = cosine(X[(cfg.FEAT_DIMS[-1]+4):], Y[(cfg.FEAT_DIMS[-1]+4):])
#     return crop_dist + full_dist

def cus_distance(X, Y, **kwargs):
    xbb = X[:4]; ybb = Y[:4]
    if ds_utils.xywh_IoU(xbb, ybb) < cfg.TEST.IOU_THRESHOLD:
        return np.finfo(np.float32).max
    return cosine(X[4:], Y[4:])


class RetrievalCNN(object):
    def __init__(self, output_dir):
        super(RetrievalCNN, self).__init__()
        
        self.output_dir  = output_dir
        base_model = ResNet50(weights='imagenet', include_top=False)
        self.model = Model(input=base_model.input, output=base_model.get_layer(cfg.LAYER).output)

    def build_search_tree(self, ctxdb, mode = 0):
        num_samples = len(ctxdb)

        if mode == 0:
            # hybrid mode
            X = np.zeros((num_samples, 4 + 2 * cfg.FEAT_DIMS[-1]))
        else:
            X = np.zeros((num_samples, 4 + cfg.FEAT_DIMS[-1]))

        for i in range(num_samples):
            ctx = ctxdb[i]
            box = ctx['box'].copy()
            xywh = ds_utils.xyxy_to_xywh(box.reshape((1,4))).squeeze().astype(np.float)

            with open(ctx['crop_feat'], 'rb') as fid:
                crop_feat = cPickle.load(fid).flatten()

            with open(ctx['full_feat'], 'rb') as fid:
                full_feat = cPickle.load(fid).flatten()

            if mode == 0:
                X[i, :] = np.concatenate((xywh, crop_feat, full_feat))
            elif mode == 1:
                X[i, :] = np.concatenate((xywh, crop_feat))
            else:
                X[i, :] = np.concatenate((xywh, full_feat))

        # if mode == 0:
        #     return BallTree(X, leaf_size=30, metric=cus_distance_hybrid)
        # else:
        #     return BallTree(X, leaf_size=30, metric=cus_distance)
        return BallTree(X, leaf_size=30, metric=cus_distance)

    def sample(self, src_ctxdb, dst_ctxdb, K, mode=0, show_gt=False):

        # Create output directories
        comp_dir = osp.join(self.output_dir, 'composite_colors')
        mask_dir = osp.join(self.output_dir, 'composite_masks')
        ds_utils.maybe_create(comp_dir)
        ds_utils.maybe_create(mask_dir)

        # Build ball tree
        dst_tree = self.build_search_tree(dst_ctxdb, mode)
        # Retrieval
        for i in xrange(len(src_ctxdb)):
            src_ctx = src_ctxdb[i]
            cand_list = self.inference_ctx(src_ctx, mode, dst_tree, K)

            # composition
            for j in range(len(cand_list)):
                dst_index = cand_list[j][0]
                dst_dist  = cand_list[j][1]
                # composition
                dst_ctx = dst_ctxdb[dst_index]
                composite_image, composite_mask = self.alpha_compose(src_ctx, dst_ctx)
                if show_gt:
                    src_img = cv2.imread(src_ctx['bg_image'], cv2.IMREAD_COLOR)
                    # dst_img = cv2.imread(dst_ctx['image'], cv2.IMREAD_COLOR)
                    # dst_img = cv2.resize(dst_img, (src_img.shape[1], src_img.shape[0]))
                    composite_image = np.concatenate((src_img, composite_image), axis=1)

                im_name, im_ext = osp.splitext(osp.basename(src_ctx['bg_image']))
                rank = src_ctx['rank']
                file_name = im_name+'_%02d'%rank + '_%02d'%j+'_'+str(dst_ctx['obj_id']).zfill(12)+im_ext
                
                output_path = osp.join(comp_dir, file_name)
                cv2.imwrite(output_path, composite_image)
                output_path = osp.join(mask_dir, file_name)
                cv2.imwrite(output_path, (composite_mask*255).astype(np.uint8))

            print i

    def alpha_compose(self, src_ctx, dst_ctx):
        # Assume src_ctx has fields: bg_image, box
        # Assume dst_ctx is from val_imdb
        src_img   = cv2.imread(src_ctx['bg_image'], cv2.IMREAD_COLOR)
        dst_img   = cv2.imread(dst_ctx['image'],    cv2.IMREAD_COLOR)
        dst_alpha = cv2.imread(dst_ctx['alpha'],    cv2.IMREAD_GRAYSCALE)

        src_xyxy  = src_ctx['box'];   dst_xyxy = dst_ctx['box']
        src_width = src_img.shape[1]; src_height = src_img.shape[0]
        dst_width = dst_img.shape[1]; dst_height = dst_img.shape[0]

        # resize the target image to align the heights of the bboxes
        factor = float(src_xyxy[3] - src_xyxy[1] + 1)/float(dst_xyxy[3] - dst_xyxy[1] + 1)
        dst_width = int(dst_width  * factor); dst_height = int(dst_height * factor)
        dst_img   = cv2.resize(dst_img, (dst_width, dst_height))
        dst_alpha = cv2.resize(dst_alpha, (dst_width, dst_height))
        dst_alpha = dst_alpha.astype(np.float)/255.0
        dst_xyxy  = factor * dst_xyxy
        src_xywh = ds_utils.xyxy_to_xywh(src_xyxy.reshape((1,4))).squeeze()
        dst_xywh = ds_utils.xyxy_to_xywh(dst_xyxy.reshape((1,4))).squeeze()

        # anchors that should match (the standing points)
        src_anchor = src_xywh[:2]; dst_anchor = dst_xywh[:2]
        offset = (src_anchor - dst_anchor).astype(np.int)

        # dilate the target patch a bit to include the blending region
        dst_bb = ds_utils.expand_xyxy(dst_xyxy.reshape((1,4)), dst_width, dst_height, ratio=0.2).squeeze().astype(np.int)
        
        src_bb = dst_bb.copy()
        src_bb[:2] = dst_bb[:2] + offset
        src_bb[2:] = dst_bb[2:] + offset

        # in case the bbox of the target object is beyond the boundaries of the source image
        if src_bb[0] < 0:
            dst_bb[0] -= src_bb[0]; src_bb[0] = 0
        if src_bb[1] < 0:
            dst_bb[1] -= src_bb[1]; src_bb[1] = 0
        if src_bb[2] > src_width - 1:
            dst_bb[2] -= src_bb[2] - src_width + 1; src_bb[2] = src_width - 1
        if src_bb[3] > src_height - 1:
            dst_bb[3] -= src_bb[3] - src_height + 1; src_bb[3] = src_height - 1


        output_mask  = np.zeros((src_height, src_width), dtype=np.float)
        output_image = src_img.copy()

        alpha_patch = dst_alpha[dst_bb[1]:(dst_bb[3]+1), dst_bb[0]:(dst_bb[2]+1)]
        src_patch   = src_img[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1),:]
        dst_patch   = dst_img[dst_bb[1]:(dst_bb[3]+1), dst_bb[0]:(dst_bb[2]+1),:]

        output_mask[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1)] = alpha_patch
        output_image[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1),:] = \
            np.expand_dims(1.0 - alpha_patch, axis=-1) * src_patch + \
            np.expand_dims(alpha_patch, axis=-1) * dst_patch

        # cv2.rectangle(output_image, (src_xyxy[0], src_xyxy[1]), (src_xyxy[2], src_xyxy[3]), \
        #                         (255, 0, 0), 1)

        return output_image.astype(np.uint8), output_mask

    def inference_ctx(self, ctx, mode, ctx_tree, K):
        # Assume ctx has fields: bg_image, box
        full_resolution = cfg.RETRIEVAL_RESOLUTION
        crop_resolution = [full_resolution[0]/2, full_resolution[1]/2, full_resolution[2]]

        box = ctx['box'].copy().astype(np.int)
        img = cv2.imread(ctx['bg_image'], cv2.IMREAD_COLOR)
        # if ctx.get('flipped', False):
        #     img = cv2.flip(img, 1)

        full_img  = cv2.resize(img, (full_resolution[1], full_resolution[0]))
        full_img  = np.expand_dims(full_img, axis=0) - cfg.PIXEL_MEANS.reshape((1,1,1,3))
        full_feat = self.model.predict(full_img).flatten()
            
        # img[box[1]:(box[3] + 1), box[0]:(box[2] + 1), :] = cfg.PIXEL_MEANS.reshape((1,1,3))
        crop_img = ds_utils.crop_and_resize(img, box.astype(np.float), full_resolution, crop_resolution)
        crop_img = np.expand_dims(crop_img, axis=0) - cfg.PIXEL_MEANS.reshape((1,1,1,3))
        crop_feat = self.model.predict(crop_img).flatten()

        xywh = ds_utils.xyxy_to_xywh(box.reshape((1,4))).squeeze().astype(np.float)

        if mode == 0:
            feat = np.concatenate((xywh, crop_feat, full_feat))
        elif mode == 1:
            feat = np.concatenate((xywh, crop_feat))
        else:
            feat = np.concatenate((xywh, full_feat))

        return self.inference_feature(feat, ctx_tree, K)

    def inference_feature(self, feat, ctx_tree, K=3):
        feat = np.expand_dims(feat.flatten(), axis=0)
        dists, indices = ctx_tree.query(feat, k=K)
        return zip(indices.flatten(), dists.flatten())


if __name__ == '__main__':
    pass
