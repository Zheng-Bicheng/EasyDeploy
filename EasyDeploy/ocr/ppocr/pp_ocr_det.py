from EasyDeploy.base import RKNNModel
import cv2
import numpy as np
from EasyDeploy.utils import (det_resize, pad_stride)
from shapely.geometry import Polygon
import pyclipper
import copy


def print_standard(standard_txt):
    print("Standard[EasyDeploy/ocr/pp_ocr/pp_ocr_det.py]: " + standard_txt)


def print_error(error_txt):
    print("Error[EasyDeploy/ocr/pp_ocr/pp_ocr_det.py]: " + error_txt)


def print_warning(warning_txt):
    print("Warning[EasyDeploy/ocr/pp_ocr/pp_ocr_det.py]: " + warning_txt)


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])


def order_points_clockwise(pts):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most

    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect


def box_score_fast(bitmap, _box):
    """
    box_score_fast: use bbox mean score as the mean score
    """
    h, w = bitmap.shape[:2]
    box = _box.copy()
    x_min = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    x_max = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    y_min = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    y_max = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - x_min
    box[:, 1] = box[:, 1] - y_min
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[y_min:y_max + 1, x_min:x_max + 1], mask)[0]


def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


def preprocess_boxes(dt_boxes, ori_im):
    def get_rotate_crop_image(img, points):
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def sorted_boxes(dt_boxes):
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

    img_crop_list = []
    dt_boxes = sorted_boxes(dt_boxes)
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)
    return dt_boxes, img_crop_list


class PPOCRDet(RKNNModel):
    def __init__(self,
                 verbose=True,
                 device=None,
                 mean_values=None,
                 std_values=None,
                 target_platform=None,
                 model_path=None,
                 input_size=None,
                 thresh=0.5,
                 box_thresh=0.5):
        # config device
        assert (device is not None) and (device.lower() in ['pc', 'board']), \
            print_error("请输入正确的设备型号(pc,board)")
        super(PPOCRDet, self).__init__(
            verbose=verbose,
            device=device
        )

        # create model
        if mean_values is None:
            mean_values = [[round(std * 255, 3) for std in [0.485, 0.456, 0.406]]]
        if std_values is None:
            std_values = [[round(mean * 255, 3) for mean in [0.229, 0.224, 0.225]]]
        assert model_path is not None, print_error("model_path is None")
        self.create_model(
            mean_values=mean_values,
            std_values=std_values,
            target_platform=target_platform,
            model_path=model_path)

        if input_size is None:
            self.input_size = (960, 960)
        self.max_candidates = 1000
        self.unclip_ratio = 2.0
        self.min_size = 3
        self.score_mode = "fast"
        self.use_dilation = False
        self.thresh = thresh
        self.box_thresh = box_thresh

    def detect(self,
               img):
        # get input image
        src_image = img.copy()
        im, ratio = det_resize(img.copy(), self.input_size)
        im = pad_stride(im, self.input_size)
        im = np.expand_dims(im, axis=0)

        # infer
        results = self.infer([im])

        # detect
        result = results[0]
        pred = result[:, 0, :, :]

        segmentation = pred > self.thresh
        src_h, src_w = self.input_size
        mask = segmentation[0]
        boxes, scores = self.boxes_from_bitmap(pred[0], mask, src_w, src_h)
        dt_boxes = self.filter_tag_det_res(boxes)
        dt_boxes, img_crop_list = preprocess_boxes(dt_boxes, src_image)
        return dt_boxes, img_crop_list

    def filter_tag_det_res(self, dt_boxes):
        img_height, img_width = self.input_size[0], self.input_size[1]
        dt_boxes_new = []
        for box in dt_boxes:
            box = order_points_clockwise(box)
            box = clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """
        contours = None
        bitmap = _bitmap
        height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print(outs)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = box_score_fast(pred, points.reshape(-1, 2))
            # print(score)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape((-1, 1, 2))
            box, sside = get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded
