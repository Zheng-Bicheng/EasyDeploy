from EasyDeploy.base.base_model import RKNNModel
import cv2
import numpy as np


def print_standard(standard_txt):
    print("Standard[EasyDeploy/detection/scrfd/scrfd.py]: " + standard_txt)


def print_error(error_txt):
    print("Error[EasyDeploy/detection/scrfd/scrfd.py]: " + error_txt)


def print_warning(warning_txt):
    print("Warning[EasyDeploy/detection/scrfd/scrfd.py]: " + warning_txt)


def draw_box(img, results, class_label, scale_x, scale_y):
    label_list = list(
        # 改动点
        map(lambda x: x.strip(), class_label))
    for i in range(len(results)):
        bbox = results[i, 2:]
        label_id = int(results[i, 0])
        score = results[i, 1]
        if score > 0.20:
            x_min, y_min, x_max, y_max = [
                int(bbox[0] * scale_x), int(bbox[1] * scale_y),
                int(bbox[2] * scale_x), int(bbox[3] * scale_y)
            ]
            print(scale_x, scale_y)
            print(x_min, y_min, x_max, y_max)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            # cv2.imshow("name", img)
            # cv2.waitKey(0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_text = label_list[label_id]
            print("label: {} ,pred:{}, loc:(min:{},max:{})".format(label_text, str(round(score, 3)), (x_min, y_min),
                                                                   (x_max, y_max)))
            img = cv2.putText(img, label_text, (x_min, y_min - 40), font, 1,
                              (0, 0, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, str(round(score, 3)), (x_min, y_min), font, 1,
                              (0, 0, 0), 1, cv2.LINE_AA)
    return img


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


# 计算IoU，矩形框的坐标形式为xyxy
def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(
                current_box, axis=0), )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def warp_boxes(boxes, ori_shape):
    """Apply transform to boxes
    """
    width, height = ori_shape[1], ori_shape[0]
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return


def softmax(m):
    shape = m.shape[0]
    m_row_max = m.max(axis=1)
    m = m - m_row_max.reshape(shape, 1)
    m_exp = np.exp(m)
    m_exp_row_sum = m_exp.sum(axis=1).reshape(shape, 1)
    result = m_exp / m_exp_row_sum
    return result


class Picodet(RKNNModel):
    def __init__(self,
                 verbose=True,
                 device=None,
                 mean_values=None,
                 std_values=None,
                 target_platform=None,
                 model_path=None,
                 input_size=None,
                 nms_top_k=1000,
                 keep_top_k=100,
                 score_threshold=0.5,
                 nms_threshold=0.5):
        # config device
        assert (device is not None) and (device.lower() in ['pc', 'board']), \
            print_error("请输入正确的设备型号(pc,board)")
        super(Picodet, self).__init__(
            verbose=verbose,
            device=device
        )

        # create model
        if mean_values is None:
            mean_values = [[round(std * 255, 3) for std in [0.229, 0.224, 0.225]]]
        if std_values is None:
            std_values = [[round(mean * 255, 3) for mean in [0.485, 0.456, 0.406]]]
        assert model_path is not None, print_error("model_path is None")
        self.create_model(
            mean_values=mean_values,
            std_values=std_values,
            target_platform=target_platform,
            model_path=model_path)

        if input_size is None:
            self.input_size = [320, 320]
        self.is_scale = True
        self.strides = [8, 16, 32, 64]

        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def resize(self, im, im_info, interp=cv2.INTER_CUBIC):
        target_size = self.input_size
        assert len(target_size) == 2
        assert target_size[0] > 0 and target_size[1] > 0
        origin_shape = im.shape[:2]
        resize_h, resize_w = target_size
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array([im_scale_y, im_scale_x]).astype('float32')
        return im, im_info

    def get_inputs(self, im):
        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': None,
        }
        if isinstance(im, str):
            src_image = cv2.imread(im)
            im = cv2.imread(im)
        else:
            src_image = im
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im, _ = self.resize(im, im_info)
        return im, src_image

    def detect(self, image, label_list):
        input_data, src_image = self.get_inputs(image)
        out_boxes_list, out_boxes_num = self.forward(input_data)
        scale_x = src_image.shape[1] / self.input_size[1]
        scale_y = src_image.shape[0] / self.input_size[0]

        res_image = draw_box(src_image, out_boxes_list, label_list, scale_x, scale_y)
        return res_image

    def forward(self, input_data):
        input_data = np.array(input_data).astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        results = self.infer([input_data])

        scores = []
        raw_boxes = []
        num_outs = int(len(results) / 2)
        for out_idx in range(num_outs):
            scores.append(results[out_idx])
            raw_boxes.append(results[out_idx + num_outs])

        test_im_shape = np.array([[self.input_size[0], self.input_size[1]]]).astype('float32')
        test_scale_factor = np.array([[1, 1]]).astype('float32')
        batch_size = raw_boxes[0].shape[0]
        reg_max = int(raw_boxes[0].shape[-1] / 4 - 1)
        out_boxes_num = []
        out_boxes_list = []
        for batch_id in range(batch_size):
            # generate centers
            decode_boxes = []
            select_scores = []
            for stride, box_distribute, score in zip(self.strides, raw_boxes, scores):
                box_distribute = box_distribute[batch_id]
                score = score[batch_id]
                # centers
                fm_h = self.input_size[0] / stride
                fm_w = self.input_size[1] / stride
                h_range = np.arange(fm_h)
                w_range = np.arange(fm_w)
                ww, hh = np.meshgrid(w_range, h_range)
                ct_row = (hh.flatten() + 0.5) * stride
                ct_col = (ww.flatten() + 0.5) * stride
                center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

                # box distribution to distance
                reg_range = np.arange(reg_max + 1)
                box_distance = box_distribute.reshape((-1, reg_max + 1))
                box_distance = softmax(box_distance)
                box_distance = box_distance * np.expand_dims(reg_range, axis=0)
                box_distance = np.sum(box_distance, axis=1).reshape((-1, 4))
                box_distance = box_distance * stride

                # top K candidate
                topk_idx = np.argsort(score.max(axis=1))[::-1]
                topk_idx = topk_idx[:self.nms_top_k]

                center = center[topk_idx]
                score = score[topk_idx]
                box_distance = box_distance[topk_idx]

                # decode box
                decode_box = center + [-1, -1, 1, 1] * box_distance

                select_scores.append(score)
                decode_boxes.append(decode_box)

            # nms
            bboxes = np.concatenate(decode_boxes, axis=0)
            confidences = np.concatenate(select_scores, axis=0)
            # print(bboxes)
            picked_box_probs = []
            picked_labels = []
            for class_index in range(0, confidences.shape[1]):
                probs = confidences[:, class_index]
                mask = probs > self.score_threshold
                probs = probs[mask]
                if probs.shape[0] == 0:
                    continue
                subset_boxes = bboxes[mask, :]
                box_probs = np.concatenate(
                    [subset_boxes, probs.reshape(-1, 1)], axis=1)
                box_probs = hard_nms(
                    box_probs,
                    iou_threshold=self.nms_threshold,
                    top_k=self.keep_top_k, )
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

            if len(picked_box_probs) == 0:
                out_boxes_list.append(np.empty((0, 4)))
                out_boxes_num.append(0)

            else:
                picked_box_probs = np.concatenate(picked_box_probs)

                # resize output boxes
                picked_box_probs[:, :4] = warp_boxes(
                    picked_box_probs[:, :4], test_im_shape[batch_id])
                im_scale = np.concatenate([
                    test_scale_factor[batch_id][::-1],
                    test_scale_factor[batch_id][::-1]
                ])
                picked_box_probs[:, :4] /= im_scale
                # clas score box
                out_boxes_list.append(np.concatenate([np.expand_dims(np.array(picked_labels), axis=-1),
                                                      np.expand_dims(picked_box_probs[:, 4], axis=-1),
                                                      picked_box_probs[:, :4]
                                                      ], axis=1))
                out_boxes_num.append(len(picked_labels))

        out_boxes_list = np.concatenate(out_boxes_list, axis=0)
        out_boxes_num = np.asarray(out_boxes_num).astype(np.int32)
        return out_boxes_list, out_boxes_num

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
            for i, _ in enumerate(results):
                results[i] = np.squeeze(results[i], 3)
        return results
