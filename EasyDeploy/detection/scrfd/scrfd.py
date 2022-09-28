from EasyDeploy.base import RKNNModel
import cv2
import numpy as np


def print_standard(standard_txt):
    print("Standard[EasyDeploy/detection/scrfd/scrfd.py]: " + standard_txt)


def print_error(error_txt):
    print("Error[EasyDeploy/detection/scrfd/scrfd.py]: " + error_txt)


def print_warning(warning_txt):
    print("Warning[EasyDeploy/detection/scrfd/scrfd.py]: " + warning_txt)


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


def nms(det_results, nms_thresh):
    thresh = nms_thresh
    x1 = det_results[:, 0]
    y1 = det_results[:, 1]
    x2 = det_results[:, 2]
    y2 = det_results[:, 3]
    scores = det_results[:, 4]

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

        index = np.where(ovr <= thresh)[0]
        order = order[index + 1]

    return keep


class SCRFD(RKNNModel):
    def __init__(self,
                 verbose=True,
                 device=None,
                 mean_values=None,
                 std_values=None,
                 target_platform=None,
                 model_path=None,
                 input_size=None):
        # config device
        assert (device is not None) and (device.lower() in ['pc', 'board']), \
            print_error("请输入正确的设备型号(pc,board)")
        super(SCRFD, self).__init__(
            verbose=verbose,
            device=device
        )

        # create model
        if mean_values is None:
            mean_values = [[round(std * 255, 3) for std in [0.5, 0.5, 0.5]]]
        if std_values is None:
            std_values = [[round(mean * 255, 3) for mean in [0.5, 0.5, 0.5]]]
        assert model_path is not None, print_error("model_path is None")
        self.create_model(
            mean_values=mean_values,
            std_values=std_values,
            target_platform=target_platform,
            model_path=model_path)

        if input_size is None:
            self.input_size = [640, 640]
        self.center_cache = {}
        self.nms_thresh = None
        self.batched = False
        self.fmc = None
        self._feat_stride_fpn = None
        self._num_anchors = None
        self.use_kps = None

    def infer(self, input_data):
        """
        推理，一般需要根据实际需求重构
        Args:
            input_data (list): 输入的数据
        Returns:
            result (list of numpy.ndarray): numpy数组
        """
        results = None
        if self.device == 'pc':
            results = self.model.inference(input_data)
        elif self.device == 'board':
            results = self.model.inference(input_data)
            for i in range(len(results)):
                results[i] = np.squeeze(results[i], 3)
        return results

    def forward(self, input_data, score_thresh):
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
                else:
                    points_predicts = None
            # If model doesn't support batching take output as is
            else:
                scores = results[idx]
                bbox_predicts = results[idx + fmc]
                bbox_predicts = bbox_predicts * stride
                if self.use_kps:
                    points_predicts = results[idx + fmc * 2] * stride
                else:
                    points_predicts = None
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

            pos_indexes = np.where(scores >= score_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_predicts)
            pos_scores = scores[pos_indexes]
            pos_bboxes = bboxes[pos_indexes]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                points = distance2kps(anchor_centers, points_predicts)
                points = points.reshape((points.shape[0], -1, 2))
                pos_points = points[pos_indexes]
                points_list.append(pos_points)
        return scores_list, bboxes_list, points_list

    def detect(self,
               img,
               score_thresh=0.5,
               nms_thresh=0.5,
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

        scores_list, bboxes_list, points_list = self.forward(det_img, score_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            points = np.vstack(points_list) / det_scale
        else:
            points = None

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det, nms_thresh)
        det = pre_det[keep, :]
        if self.use_kps:
            points = points[order, :, :]
            points = points[keep, :, :]

        if 0 < max_num < det.shape[0]:
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

    def init_vars(self, outputs):
        if len(outputs[0].shape) == 3:
            self.batched = True
        else:
            self.batched = False
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
