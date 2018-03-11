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

import keras
from keras.layers import Input, Dense, merge
from keras.layers import RepeatVector, Permute
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Reshape, Activation
from keras.layers import GlobalMaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, RMSprop
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import TensorBoard
from keras.applications.resnet50 import conv_block


class PredictionCNN(object):
    def __init__(self, output_dir):
        super(PredictionCNN, self).__init__()

        self.output_dir = output_dir
        self.cen_dims  = cfg.GRID_SHAPE[0] * cfg.GRID_SHAPE[1]
        self.size_dims = cfg.GRID_SHAPE[2] * cfg.GRID_SHAPE[3]

        self.model = self.training_model()
        self.center_inference = self.center_inference_model(self.model)
        self.size_inference   = self.size_inference_model(self.model)

        self.palette = [(0,255,0),(255,0,0),(0,0,255)]
    
    def get_minibatch(self, imdb, vis=False):
        batch_size = cfg.TRAIN.BATCH_SIZE
        resolution = cfg.PREDICT_RESOLUTION
        grid_shape = cfg.GRID_SHAPE

        images, layouts, boxes, grids = imdb.get_background_minibatch()
        rois = ds_utils.centers_to_rois(grids[:,0], grid_shape[:2], grid_shape[:2])

        cens_onehot  = to_categorical(grids[:,0], self.cen_dims)
        sizes_onehot = to_categorical(grids[:,1], self.size_dims)

        if vis:
            output_dir = osp.join(self.output_dir, 'minibatch')
            ds_utils.maybe_create(output_dir)

            for i in xrange(batch_size):

                img = images[i].copy()
                lyo = layouts[i].copy()

                cen_id  = np.argmax(cens_onehot[i,:])
                size_id = np.argmax(sizes_onehot[i,:])

                true_xywh = boxes[i, :]
                true_xywh = ds_utils.denormalize_xywh(true_xywh.reshape((1,4)), resolution[1], resolution[0])
                true_xyxy = ds_utils.xywh_to_xyxy(true_xywh, resolution[1], resolution[0]).squeeze()

                grid_xywh = ds_utils.indices_to_boxes(\
                            np.array([cen_id, size_id]).reshape((1,2)), \
                            grid_shape)
                grid_xywh = ds_utils.denormalize_xywh(grid_xywh, resolution[1], resolution[0])
                grid_xyxy = ds_utils.xywh_to_xyxy(grid_xywh, resolution[1], resolution[0]).squeeze()


                cv2.rectangle(img, (true_xyxy[0], true_xyxy[1]), (true_xyxy[2], true_xyxy[3]), \
                            (0, 255, 0), 1)
                cv2.rectangle(img, (grid_xyxy[0], grid_xyxy[1]), (grid_xyxy[2], grid_xyxy[3]), \
                            (255, 0, 0), 1)

                cv2.rectangle(lyo, (true_xyxy[0], true_xyxy[1]), (true_xyxy[2], true_xyxy[3]), \
                            (0, 255, 0), 1)
                cv2.rectangle(lyo, (grid_xyxy[0], grid_xyxy[1]), (grid_xyxy[2], grid_xyxy[3]), \
                            (255, 0, 0), 1)

                roi = rois[i].copy()
                roi = cv2.resize((roi*255).astype(np.uint8), (resolution[1], resolution[0]))

                output_path = osp.join(output_dir, 'img_%06d.jpg'%i)
                cv2.imwrite(output_path, img)

                output_path = osp.join(output_dir, 'lyo_%06d.jpg'%i)
                cv2.imwrite(output_path, lyo)

                output_path = osp.join(output_dir, 'roi_%06d.jpg'%i)
                cv2.imwrite(output_path, roi)

        return images, layouts, rois, cens_onehot, sizes_onehot

    def preprocess_input_images(self, batch_images, batch_layouts):
        batch_blurred = batch_images.copy()

        for i in range(batch_images.shape[0]):
            batch_blurred[i] = cv2.GaussianBlur(batch_images[i], (19, 19), 0)
            # output_path = '%06d.jpg'%i
            # cv2.imwrite(output_path, batch_blurred[i])

        new_images  = batch_blurred - cfg.PIXEL_MEANS.reshape((1,1,1,3))
        new_layouts = batch_layouts - cfg.PIXEL_MEANS.reshape((1,1,1,3))
        return new_images, new_layouts

    def training_model(self):
        print('Building training model...')

        resolution = cfg.PREDICT_RESOLUTION
        grid_shape = cfg.GRID_SHAPE
        state_dims = cfg.STATE_DIMS

        # Input images
        input_imgs = Input(shape=(resolution[0], resolution[1], resolution[2]), name='input_imgs')
        input_lyos = Input(shape=(resolution[0], resolution[1], resolution[2]), name='input_layouts')
        # Intermediate ROI maps for size prediction
        input_rois = Input(shape=(state_dims[0], state_dims[1]), name='input_rois')

        # Merge
        inputs = keras.layers.concatenate([input_imgs, input_lyos], axis = -1)

        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(inputs)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = conv_block(x, 3, [64, 64, 128],   stage=2,  block='a')
        x = conv_block(x, 3, [64, 64, 128],   stage=3,  block='a')
        x = conv_block(x, 3, [128, 128, 512], stage=4,  block='a')
        feats = x

        # center branch
        cen_hidden = Conv2D(64, (3, 3), dilation_rate=2, padding='same', activation='relu')(feats)
        cen_hidden = Conv2D(1,  (3, 3), dilation_rate=2, padding='same')(cen_hidden)
        cen_hidden = Reshape((self.cen_dims, ))(cen_hidden)
        cen_output = Activation('softmax', name = 'output_cens')(cen_hidden)

        # size branch
        rois = Reshape((state_dims[0] * state_dims[1], ))(input_rois)
        rois = RepeatVector(state_dims[2])(rois)
        rois = Permute((2, 1))(rois)
        rois = Reshape((state_dims[0], state_dims[1], state_dims[2]))(rois)

        size_hidden = Conv2D(state_dims[2], (3, 3), dilation_rate=2, padding='same', activation='relu')(feats)
        size_hidden = keras.layers.multiply([size_hidden, rois])
        size_hidden = GlobalMaxPooling2D()(size_hidden)
        size_hidden = Dense(self.size_dims, activation = 'relu')(size_hidden)
        size_output = Dense(self.size_dims, activation = 'softmax', name = 'output_sizes')(size_hidden)

        model = Model(inputs  = [input_imgs, input_lyos, input_rois], \
                      outputs = [cen_output, size_output])


        model.compile(loss = {'output_cens':  'categorical_crossentropy', \
                              'output_sizes': 'categorical_crossentropy'}, \
                      loss_weights = {'output_cens':  1.0, \
                                      'output_sizes': 2.0}, \
                      optimizer = Adam(lr=cfg.TRAIN.INITIAL_LR, \
                                          clipnorm=cfg.TRAIN.CLIP_GRADIENTS), \
                      metrics = {'output_cens':  'categorical_accuracy', \
                                 'output_sizes': 'categorical_accuracy'})

        # print(model.summary())
        # print(model.metrics_names)
        # for x in model.get_weights():
        #     print x.shape
        return model

    def center_inference_model(self, trained_model=None):
        print('Building center inference model')

        resolution = cfg.PREDICT_RESOLUTION
        grid_shape = cfg.GRID_SHAPE
        state_dims = cfg.STATE_DIMS

        ########################################################################
        # Input
        ########################################################################
        input_img = Input(batch_shape=(1, resolution[0], resolution[1], resolution[2]), name='input_img')
        input_lyo = Input(batch_shape=(1, resolution[0], resolution[1], resolution[2]), name='input_layout')

        # Merge
        inputs = keras.layers.concatenate([input_img, input_lyo], axis = -1)

        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(inputs)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = conv_block(x, 3, [64, 64, 128],   stage=2,  block='a')
        x = conv_block(x, 3, [64, 64, 128],   stage=3,  block='a')
        x = conv_block(x, 3, [128, 128, 512], stage=4,  block='a')

        feat = x

        # center branch
        cen_hidden = Conv2D(64, (3, 3), dilation_rate=2, padding='same', activation='relu')(feat)
        cen_hidden = Conv2D(1,  (3, 3), dilation_rate=2, padding='same')(cen_hidden)
        cen_hidden = Reshape((self.cen_dims, ))(cen_hidden)
        cen_output = Activation('softmax', name = 'output_cen')(cen_hidden)

        ########################################################################
        # Compile inference model
        ########################################################################
        inference_model = Model(inputs=[input_img, input_lyo], outputs=[feat, cen_output])

        # print(inference_model.summary())
        if trained_model is not None:
            inference_model.set_weights(self.get_center_branch_weights(trained_model))
        return inference_model

    def size_inference_model(self, trained_model=None):
        print('Building size inference model')

        state_dims = cfg.STATE_DIMS
        feat_dims  = cfg.STATE_DIMS#cfg.FEAT_DIMS

        ########################################################################
        # Input
        ########################################################################
        input_feat = Input(batch_shape=(1, feat_dims[0], feat_dims[1], feat_dims[2]), \
                        name='input_feat')
        input_roi  = Input(batch_shape=(1, state_dims[0], state_dims[1]), \
                        name='input_roi')

        # size branch
        rois = Reshape((state_dims[0] * state_dims[1], ))(input_roi)
        rois = RepeatVector(state_dims[2])(rois)
        rois = Permute((2, 1))(rois)
        rois = Reshape((state_dims[0], state_dims[1], state_dims[2]))(rois)

        size_hidden = Conv2D(state_dims[2], (3, 3), dilation_rate=2, padding='same', activation='relu')(input_feat)
        size_hidden = keras.layers.multiply([size_hidden, rois])
        size_hidden = GlobalMaxPooling2D()(size_hidden)
        size_hidden = Dense(self.size_dims, activation = 'relu')(size_hidden)
        size_output = Dense(self.size_dims, activation = 'softmax', name = 'output_size')(size_hidden)

        ########################################################################
        # Compile inference model
        ########################################################################
        inference_model = Model(inputs  = [input_feat, input_roi], outputs = size_output)

        # print(inference_model.summary())
        if trained_model is not None:
            inference_model.set_weights(self.get_size_branch_weights(trained_model))
        return inference_model

    def get_center_branch_weights(self, trained_model):
        # Super hacky
        trained_weights = trained_model.get_weights()
        weights = trained_weights[:-10]
        weights.extend(trained_weights[-8:-4])
        return weights

    def get_size_branch_weights(self, trained_model):
        # Super hacky
        trained_weights = trained_model.get_weights()
        weights = trained_weights[-10:-8]
        weights.extend(trained_weights[-4:])
        return weights
    
    def generator(self, imdb, batch_size):
        while 1:
            batch_size = cfg.TRAIN.BATCH_SIZE
            num_iteration = len(imdb.objdb) // batch_size
            imdb.permute_objdb_indices()
            for i in range(num_iteration):
                images, layouts, rois, cens, sizes = self.get_minibatch(imdb)
                input_imgs, input_lyos = self.preprocess_input_images(images, layouts)
                repeat_cens = np.expand_dims(cens, axis=1)
                repeat_cens = np.repeat(repeat_cens, cfg.TOP_K, axis=1)

                yield ({'input_imgs': input_imgs, 'input_lyos': input_lyos, 'input_rois': rois}, {'output_cens': repeat_cens, 'output_sizes': sizes })
    
    def sampler(self, test_db, epoch=0, K=3, vis=False):
        # assume each entry in test_db has field: 'bg_image', 'bg_layout'
        self.center_inference.set_weights(self.get_center_branch_weights(self.model))
        self.size_inference.set_weights(self.get_size_branch_weights(self.model))

        output_dir = osp.join(self.output_dir, 'prediction_jsons')
        ds_utils.maybe_create(output_dir)
        if vis:
            vis_dir = osp.join(self.output_dir, 'prediction_vis')
            ds_utils.maybe_create(vis_dir)
            # hm_dir = osp.join(self.output_dir, 'prediction_heatmap')
            # ds_utils.maybe_create(hm_dir)


        res_db = []
        num_samples = len(test_db)
        
        for i in range(num_samples):
            entry   = test_db[i]
            im_path = entry['bg_image']
            im_name, im_ext = osp.splitext(osp.basename(im_path))
            ori_img = cv2.imread(im_path, cv2.IMREAD_COLOR)
            img, ox, oy = ds_utils.create_squared_image(ori_img, cfg.PIXEL_MEANS)
            width  = img.shape[1];height = img.shape[0]

            xywhs, grids, heatmap = self.single_sample(entry,K=K)
            xywhs = ds_utils.denormalize_xywh(xywhs, width, height)
            xyxys = ds_utils.xywh_to_xyxy(xywhs, width, height)

            xyxys[:,0] -= ox; xyxys[:,1] -= oy
            xyxys[:,2] -= ox; xyxys[:,3] -= oy
            xyxys = ds_utils.clip_boxes(xyxys, ori_img.shape[1], ori_img.shape[0])
            heatmap = heatmap[oy:(oy+ori_img.shape[0]), ox:(ox+ori_img.shape[1]), :]

            res = {}
            res['bg_image'] = im_path
            res['name']  = im_name
            res['boxes'] = xyxys.tolist()
            json_path = osp.join(output_dir, im_name+'.json')
            with open(json_path, 'w') as res_file:
                json.dump(res, res_file, indent=4, separators=(',', ': '))

            if vis:
                vis_img = ori_img
                fontScale = 0.0007 * math.sqrt(2 * width * height)
                for j in range(xyxys.shape[0]):
                    bb = xyxys[j]
                    color = self.palette[j%len(self.palette)]
                    cv2.rectangle(vis_img, (bb[0], bb[1]), (bb[2], bb[3]), color, 4)
                    # cv2.putText(vis_img, '{:}'.format(j), (bb[0], bb[1] - 2),
                    #             cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1)

                tmp = np.ones_like(heatmap, dtype=np.float)
                tmp[:,:,1] += heatmap[:,:,1]/255.0
                overlay = np.multiply(vis_img, tmp)
                overlay = np.minimum(overlay, 255).astype(np.uint8)
                final = np.concatenate((vis_img, overlay, heatmap), axis=1)
                # output_path = osp.join(vis_dir, '%04d_'%epoch+im_name+im_ext)
                # cv2.imwrite(output_path, final)

                output_path = osp.join(vis_dir, '%04d_'%epoch+im_name+'_ol'+im_ext)
                cv2.imwrite(output_path, overlay)
                output_path = osp.join(vis_dir, '%04d_'%epoch+im_name+'_hm'+im_ext)
                cv2.imwrite(output_path, heatmap)

            for j in range(len(res['boxes'])):
                entry = {}
                entry['bg_image'] = im_path
                entry['name']  = im_name
                entry['box']   = xyxys[j]
                entry['rank']  = j
                res_db.append(entry)
        
        return res_db

    def train(self, train_imdb, val_imdb, test_db):
        ckpt_dir = osp.join(self.output_dir, 'prediction_ckpts')
        ds_utils.maybe_create(ckpt_dir)
        ckpt_path = osp.join(ckpt_dir, "weights.{epoch:02d}-{val_output_cens_loss:.2f}-{val_output_sizes_loss:.2f}-{val_output_cens_mcl_accu:.2f}-{val_output_sizes_categorical_accuracy:.2f}.hdf5")

        log_dir = osp.join(self.output_dir, 'prediction_logs')
        ds_utils.maybe_create(log_dir)

        checkpointer = ModelCheckpoint(filepath=ckpt_path, verbose=1, save_weights_only=False)
        logwriter = TensorBoard(log_dir=log_dir)

        vis_sampler = LambdaCallback(
            on_epoch_end=lambda epoch, logs: self.sampler(test_db, epoch, vis=True))

        self.model.fit_generator(
            self.generator(train_imdb, cfg.TRAIN.BATCH_SIZE),
            steps_per_epoch=int(len(train_imdb.objdb)/cfg.TRAIN.BATCH_SIZE),
            epochs=cfg.TRAIN.EPOCH, callbacks=[checkpointer, logwriter, vis_sampler],
            validation_data=self.generator(val_imdb, cfg.TRAIN.BATCH_SIZE),
            validation_steps=int(len(val_imdb.objdb)/cfg.TRAIN.BATCH_SIZE))

    def train_old(self, train_imdb, val_imdb):
        ###########################################################################
        """Load checkpoint if available"""
        if self.load_checkpoint():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        ###########################################################################
        batch_size = cfg.TRAIN.BATCH_SIZE
        num_iteration = len(train_imdb.objdb) // batch_size
        step = 0
        for epoch in xrange(cfg.TRAIN.EPOCH):
            train_imdb.permute_objdb_indices()

            for i in xrange(num_iteration):
                images, layouts, rois, cens, sizes = self.get_minibatch(train_imdb)
                input_imgs, input_lyos = self.preprocess_input_images(images, layouts)

                repeat_cens = np.expand_dims(cens, axis=1)
                repeat_cens = np.repeat(repeat_cens, cfg.TOP_K, axis=1)

                # one iteration
                log = self.model.train_on_batch(\
                    {'input_imgs': input_imgs, 'input_lyos': input_lyos, 'input_rois': rois}, \
                    {'output_cens': repeat_cens, 'output_sizes': sizes })

                # verboses
                if ((step + 1) % 10 == 0) or i == num_iteration - 1:
                    print("Iter %d: loss-%f, cen_loss-%f, size_loss-%f, cen_accu-%f, size_accu-%f" % \
                        (step+1, log[0], log[1], log[2], log[3], log[4]))

                # validation
                if i == num_iteration - 1:
                    log = self.validation(val_imdb)
                    print("Epoch %d: loss-%f, cen_loss-%f, size_loss-%f, cen_accu-%f, size_accu-%f" % \
                        (epoch+1, log[0], log[1], log[2], log[3], log[4]))
                    self.save_checkpoint(epoch+1, log)

                # samples
                    self.center_inference.set_weights(self.get_center_branch_weights(self.model))
                    self.size_inference.set_weights(self.get_size_branch_weights(self.model))
                    self.batch_sample(train_imdb.objdb, epoch+1)

                step+=1

    def single_sample(self, entry, K=3):
        # assuming square image
        resolution = cfg.PREDICT_RESOLUTION
        grid_shape = cfg.GRID_SHAPE
        state_dims = cfg.STATE_DIMS

        img = cv2.imread(entry['bg_image'],  cv2.IMREAD_COLOR)
        lyo = cv2.imread(entry['bg_layout'], cv2.IMREAD_COLOR)
        img, ox, oy = ds_utils.create_squared_image(img, cfg.PIXEL_MEANS)
        lyo, ox, oy = ds_utils.create_squared_image(lyo, cfg.PIXEL_MEANS)
        input_img = cv2.resize(img, (resolution[1], resolution[0]))
        input_lyo = cv2.resize(lyo, (resolution[1], resolution[0]))
        input_img, input_lyo = self.preprocess_input_images(\
            np.expand_dims(input_img, 0), np.expand_dims(input_lyo, 0))

        feat, cen_probs = self.center_inference.predict([input_img, input_lyo])
        cen_probs = cen_probs.squeeze()
        cen_inds = np.argsort(cen_probs)[::-1]
        cen_inds = np.array(cen_inds[:K])
        rois = ds_utils.centers_to_rois(cen_inds, grid_shape[:2], grid_shape[:2])

        grids = np.zeros((0,2))
        for i in range(K):
            input_roi  = rois[i].reshape((1, state_dims[0], state_dims[1]))
            size_probs = self.size_inference.predict([feat, input_roi]).squeeze()
            size_inds = np.argsort(size_probs.squeeze())[::-1]
            size_inds = np.array(size_inds[:K])
            for j in range(K):
                grids = np.vstack((grids, np.array([cen_inds[i], size_inds[j]]).reshape((1,2))))

        xywhs = ds_utils.indices_to_boxes(grids, grid_shape)
        
        cen_probs = cen_probs.reshape((grid_shape[0], grid_shape[1]))
        heatmap = cv2.resize(cen_probs, (img.shape[1], img.shape[0]))
        heatmap = (255 * heatmap).astype(np.uint8)
        heatmap = cv2.equalizeHist(heatmap)
        heatmap = np.repeat(np.expand_dims(heatmap, axis=-1), 3, axis=-1)

        return xywhs, grids, heatmap

    def load_checkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        if not osp.exists(checkpoint_dir):
            return False
        ckpt = sorted(os.listdir(checkpoint_dir))
        if len(ckpt) > 0:
            print("[*] Initializing from %s." % ckpt[-1])
            self.model.load_weights(osp.join(checkpoint_dir, ckpt[-1]))
            return True
        else:
            return False

    def save_checkpoint(self, step, log):
        print(" [*] Saving checkpoints...")
        checkpoint_dir = osp.join(self.output_dir, 'prediction_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_name = "weights-%07d-%.4f-%.4f-%.4f-%.4f.hdf5" % (step, log[1], log[2], log[3], log[4])
        self.model.save(osp.join(checkpoint_dir, model_name))


if __name__ == '__main__':
    pass