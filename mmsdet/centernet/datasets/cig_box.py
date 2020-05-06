from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import cv2

import torch.utils.data as data

from ..core.image import flip, color_aug
from ..core.image import get_affine_transform, affine_transform
from ..core.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from ..core.image import draw_dense_reg
import math


class CigBox(data.Dataset):
  num_classes = 1
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(CigBox, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'cig_box')
    self.img_dir = os.path.join(self.data_dir, 'images', '{}'.format(split))
    if split == 'val':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'val.json')
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'train.json')
      else:
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'train.json')
    self.max_objs = 128
    self.class_name = [
      '__background__', 'cigarette']
    self._valid_ids = [0,1]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing cig_box {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)
    # print('1',img.shape)
    # cv2.imwrite('./org.jpg', img)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
      # print('2', input_h, input_w, s)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
      # print('3', s, input_h, input_w)    # 1024.0 512 512
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)    # does not crop the img, only imagine the center locat
        # print('croped', w_border, h_border, c)    # 128 128 [419. 308.]
        # cv2.imwrite('./croped.jpg', )
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    # print('4',c, s,  trans_input)    # [419. 308.] 1228.7999999999997 [[ 4.16666650e-01 -0.00000000e+00  8.14166736e+01]
                                     #                                [ 3.39161210e-17  4.16666650e-01  1.27666672e+02]
    inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)    # out shape (w, h) not (h, w)
    # print('5', inp.shape)    # (512, 512, 3)
    # cv2.imwrite('./warpAffine.jpg', inp)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
      # print('color_auged')    # yes
      # cv2.imwrite('./color_auged.jpg', inp*self.std + self.mean)
      # cv2.imwrite('./color_auged.jpg', inp*255)
    # inp_ = (inp*255).copy()
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
    # print('6', c, s, trans_output)

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)    # max_0bjs can reduce from label.json
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    # print(draw_gaussian)

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      # print('org bbox', bbox)
      # x1, y1, x2, y2 = bbox    # org bbox
      # cv2.rectangle(inp_, (x1, y1), (x2, y2), (255,0,0), 2)
      # cv2.imwrite('./img_affine_color_see_bbox.jpg', inp_)
      cls_id = int(self.cat_ids[ann['category_id']]) -1     # for just one classes
      if flipped:
        # print('flipped')
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)    # affine on bbox to heatmap scale
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      # x1, y1, x2, y2 = bbox*4
      # cv2.rectangle(inp_, (x1, y1), (x2, y2), (255,0,0), 2)
      # cv2.imwrite('./img_affine_color_see_bbox.jpg', inp_)
      # print('after affine bbox', bbox)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]     
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))    # up integer
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        # print('7', self.opt.mse_loss)
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
          
        ct_int = ct.astype(np.int32)
        # print(np.sum(hm[cls_id]), 'ff')    # all is zero
        draw_gaussian(hm[cls_id], ct_int, radius)    # hm[cls_id].shape, (128, 128)
        # cv2.imwrite('./gaussian_heatmap.jpg', hm[cls_id]*255)
        # print(np.sum(hm[cls_id]), 'aff')    # a gaussian heatmap radius, max is 1.0
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]    # y_c * output_w + x_c:index of max heat on heatmap img(128, 128)
        reg[k] = ct - ct_int    # center shift
        reg_mask[k] = 1   # exist or not bbox in the max_obj vector
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]    # spec?
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          # print('dense?')
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])    # xyxy, 1, cls_id on heatmap, not org img
    
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      # print('8', self.opt.cat_spec_wh)
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    # print('xxx', ret)
    # print(hm.shape, inp.shape, reg_mask.shape, ind.shape, wh.shape)  # (1, 128, 128) (3, 512, 512) (128,) (128,) (128, 2)
    return ret

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
