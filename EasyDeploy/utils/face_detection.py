import numpy as np
import cv2


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


def draw_face(img, bboxes, kpss, out_path, with_kps=True):
    """
    绘制人脸框
    Args:
        img: 输入的人脸图片，建议为opencv读取的图片
        bboxes: 框
        kpss: 点
        out_path: 图片保存路径
        with_kps: 是否画点

    Returns: 无

    """
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1, y1, x2, y2, score = bbox.astype(np.int32)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if with_kps:
            if kpss is not None:
                kps = kpss[i].reshape(-1, 2)
                for kp in kps:
                    kp = kp.astype(np.int32)
                    cv2.circle(img, tuple(kp), 1, (255, 0, 0), 2)

    print('output:', out_path)
    cv2.imwrite(out_path, img)
