from EaysDeploy.base import RKNNModelPC
from EaysDeploy.base import RKNNModelBoard
import cv2
import numpy as np


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


class ScrFDForPC(RKNNModelPC):
    def __init__(self,
                 verbose=True,
                 mean_values=None,
                 std_values=None,
                 target_platform=None,
                 model_path=None,
                 input_size=None,
                 nms_thresh=0.5):
        if mean_values is None:
            mean_values = [[round(std * 255, 3) for std in [0.5, 0.5, 0.5]]]
        if std_values is None:
            std_values = [[round(mean * 255, 3) for mean in [0.5, 0.5, 0.5]]]
        super(ScrFDForPC, self).__init__(
            verbose=verbose,
            mean_values=mean_values,
            std_values=std_values,
            target_platform=target_platform,
            model_path=model_path
        )
        if input_size is None:
            self.input_size = [640, 640]
        self.center_cache = {}
        self.nms_thresh = nms_thresh
        self.batched = False

    def forward(self, input_data, thresh):
        scores_list = []
        bboxes_list = []
        points_list = []
        results = self.infer([input_data])
        self.init_vars(results)
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = results[idx][0]
                bbox_predicts = results[idx + fmc][0]
                bbox_predicts = bbox_predicts * stride
                if self.use_kps:
                    points_predicts = results[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = results[idx]
                bbox_predicts = results[idx + fmc]
                bbox_predicts = bbox_predicts * stride
                if self.use_kps:
                    points_predicts = results[idx + fmc * 2] * stride

            height = self.input_size[0] // stride
            width = self.input_size[1] // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_indexes = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_predicts)
            pos_scores = scores[pos_indexes]
            pos_bboxes = bboxes[pos_indexes]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                points = distance2kps(anchor_centers, points_predicts)
                # points = points_predicts
                points = points.reshape((points.shape[0], -1, 2))
                pos_points = points[pos_indexes]
                points_list.append(pos_points)
        return scores_list, bboxes_list, points_list

    def detect(self,
               img,
               thresh=0.5,
               max_num=0,
               metric='default'):
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        if im_ratio > model_ratio:
            new_height = self.input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        det_img = np.expand_dims(det_img, axis=0)
        scores_list, bboxes_list, points_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            points = np.vstack(points_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            points = points[order, :, :]
            points = points[keep, :, :]
        else:
            points = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                              det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if points is not None:
                points = points[bindex, :]
        return det, points

    # def predict(self, image):

    def init_vars(self, outputs):
        if len(outputs[0].shape) == 3:
            self.batched = True
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


class ScrFDForBoard(RKNNModelBoard):
    def __init__(self,
                 verbose=True,
                 rknn_path=None,
                 target='RK3568',
                 input_size=None,
                 nms_thresh=0.5):
        super(ScrFDForBoard, self).__init__(
            verbose=True,
            rknn_path=None,
            target='RK3568'
        )
        if input_size is None:
            self.input_size = [640, 640]
        self.center_cache = {}
        self.nms_thresh = nms_thresh
        self.batched = False

    def forward(self, input_data, thresh):
        scores_list = []
        bboxes_list = []
        points_list = []
        results = self.infer([input_data])
        self.init_vars(results)
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            # If model support batch dim, take first output
            if self.batched:
                scores = results[idx][0]
                bbox_predicts = results[idx + fmc][0]
                bbox_predicts = bbox_predicts * stride
                if self.use_kps:
                    points_predicts = results[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = results[idx]
                bbox_predicts = results[idx + fmc]
                bbox_predicts = bbox_predicts * stride
                if self.use_kps:
                    points_predicts = results[idx + fmc * 2] * stride

            height = self.input_size[0] // stride
            width = self.input_size[1] // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_indexes = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_predicts)
            pos_scores = scores[pos_indexes]
            pos_bboxes = bboxes[pos_indexes]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                points = distance2kps(anchor_centers, points_predicts)
                # points = points_predicts
                points = points.reshape((points.shape[0], -1, 2))
                pos_points = points[pos_indexes]
                points_list.append(pos_points)
        return scores_list, bboxes_list, points_list

    def detect(self,
               img,
               thresh=0.5,
               max_num=0,
               metric='default'):
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        if im_ratio > model_ratio:
            new_height = self.input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        det_img = np.expand_dims(det_img, axis=0)
        scores_list, bboxes_list, points_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            points = np.vstack(points_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            points = points[order, :, :]
            points = points[keep, :, :]
        else:
            points = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                              det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if points is not None:
                points = points[bindex, :]
        return det, points

    # def predict(self, image):

    def init_vars(self, outputs):
        if len(outputs[0].shape) == 3:
            self.batched = True
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep