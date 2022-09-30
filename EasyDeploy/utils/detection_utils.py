import numpy as np
import cv2


def draw_bbox(img, bboxes, out_path=None):
    """
    绘制方框
    Args:
        img: 输入的人脸图片，建议为opencv读取的图片
        bboxes: 框
        out_path: 图片保存路径，为空则不保存

    Returns:
        output_img: 画了方框之后的图片
    """
    output_img = None

    # rectangle
    input_img = img.copy()
    for bbox in bboxes.shape:
        x1, y1, x2, y2, score = bbox.astype(np.int32)
        output_img = cv2.rectangle(input_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # save
    if out_path is not None:
        cv2.imwrite(out_path, img)
    return output_img


def draw_key_points(img, key_points, out_path=None):
    """
    绘制关键点
    Args:
        img: 输入的人脸图片，建议为opencv读取的图片
        key_points: 关键点
        out_path: 图片保存路径，为空则不保存

    Returns:
        output_img: 画了方框之后的图片
    """
    output_img = None
    input_key_points = key_points.copy()
    for key_point in input_key_points:
        key_point = key_point.reshape(-1, 2)
        for point in key_point:
            point = point.astype(np.int32)
            output_img = cv2.circle(img, tuple(point), 1, (255, 0, 0), 2)
    # save
    if out_path is not None:
        cv2.imwrite(out_path, img)
    return output_img


def pad_stride(img, pad_shape):
    """
    从左往右填充im到指定的
    Args:
        img: mat,the shape of mat is (h,w,c)
        pad_shape: the shape of output

    Returns:
        padded picture
    """
    im_h, im_w, im_c = img.shape
    pad_h = pad_shape[0]
    pad_w = pad_shape[1]
    padding_im = np.full((pad_h, pad_w, 3), 128, np.uint8)
    padding_im[:im_h, :im_w, :im_c] = img
    return padding_im


def det_resize(img, target_size, interp=cv2.INTER_CUBIC):
    """
    resize
    Args:
        img: mat,the shape of mat is (h,w,c)
        target_size: the shape of output
        interp:

    Returns:

    """
    output_h = target_size[0]
    output_w = target_size[1]
    h, w, c = img.shape
    if h > w:
        ratio = float(output_h) / h
    elif w < h:
        ratio = float(output_w) / w
    else:
        ratio = 1
    img = cv2.resize(img, None, None, fx=ratio, fy=ratio, interpolation=interp)
    return img, ratio
