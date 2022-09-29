import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os


def get_similarity(data1, data2):
    def l2_norm(input, axis=1):
        norm = np.linalg.norm(input, 2, axis, True)
        output = np.divide(input, norm)
        return output

    data1 -= np.average(data1)
    data2 -= np.average(data2)

    data1 = l2_norm(data1)
    data2 = l2_norm(data2)
    prob = np.dot(data1, data2.T) / (np.linalg.norm(data1) * np.linalg.norm(data2))
    return prob.ravel()[0]


def norm_crop(img, landmark, image_size=112):
    """
    人脸对齐函数
    Args:
        img: 输入的人脸图片，建议为opencv读取的图片
        landmark: 五个人脸关键点，输入时建议reshape成[-1,2]
        image_size: 对齐后图片的尺寸

    Returns: 对齐后的人脸图片

    """

    def transformation_from_points(points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2
        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    img_size = np.array([112, 112])
    coord5point = np.array([(0.31556875000000000, 0.4615741071428571),
                            (0.68262291666666670, 0.4615741071428571),
                            (0.50026249999999990, 0.6405053571428571),
                            (0.34947187500000004, 0.8246919642857142),
                            (0.65343645833333330, 0.8246919642857142)])
    coord5point = coord5point * img_size
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in landmark]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in coord5point]))
    M = transformation_from_points(pts1, pts2)
    warped = cv2.warpAffine(img, M[:2], (image_size, image_size))
    return warped


class ColorMap(object):
    def __init__(self, num):
        super().__init__()
        self.get_color_map_list(num)
        self.color_map = {}
        self.ptr = 0

    def __getitem__(self, key):
        return self.color_map[key]

    def update(self, keys):
        for key in keys:
            if key not in self.color_map:
                i = self.ptr % len(self.color_list)
                self.color_map[key] = self.color_list[i]
                self.ptr += 1

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        self.color_list = [
            color_map[i:i + 3] for i in range(0, len(color_map), 3)
        ]


def draw_face_with_text(img, box_list, results, label_idx_list):
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    color_map = ColorMap(100)
    color_map.update(label_idx_list)
    font_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "simsun.ttc")
    for i, dt in enumerate(box_list):
        bbox, _ = dt[0:-1], dt[-1]
        label = results[i][0][0]
        score = results[i][0][1]
        color = tuple(color_map[label])
        xmin, ymin, xmax, ymax = bbox
        font_size = max(int((xmax - xmin) // 6), 10)
        font = ImageFont.truetype(font_path, font_size)

        text = "{} {:.4f}".format(label, score)
        th = sum(font.getmetrics())
        tw = font.getsize(text)[0]
        start_y = max(0, ymin - th)

        draw.rectangle((xmin, start_y, xmin + tw + 1, start_y + th), fill=color)
        draw.text((xmin + 1, start_y), text, fill=(255, 255, 255), font=font, anchor="la")
        draw.rectangle((xmin, ymin, xmax, ymax), width=2, outline=color)
    return np.array(im)
