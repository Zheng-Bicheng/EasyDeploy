from EasyDeploy.detection.scrfd import SCRFD
from EasyDeploy.clas.ada_face import AdaFace
import os
import cv2
import numpy as np
from EasyDeploy.utils import norm_crop
from EasyDeploy.utils import get_similarity


def print_standard(standard_txt):
    print("Standard[EasyDeploy/project/face_recognition/face_recognition.py]: " + standard_txt)


def print_error(error_txt):
    print("Error[EasyDeploy/project/face_recognition/face_recognition.py]: " + error_txt)


def print_warning(warning_txt):
    print("Warning[EasyDeploy/project/face_recognition/face_recognition.py]: " + warning_txt)


face_det_model_name_ls = ['scrfd', ]
face_cls_model_name_ls = ['adaface', ]


class FaceRecognition:
    def __init__(self,
                 det_model_name,
                 det_model_weight_path,
                 cls_model_name,
                 cls_model_weight_path,
                 verbose=True,
                 device=None,
                 target_platform="RK3568"
                 ):
        assert det_model_name.lower() in face_det_model_name_ls, "目前人脸识别检测(det)模型只支持{}".format(
            face_det_model_name_ls)
        assert cls_model_name.lower() in face_cls_model_name_ls, "目前人脸识别分类(cls)模型只支持{}".format(
            face_cls_model_name_ls)
        assert (device is not None) and (device.lower() in ['pc', 'board']), \
            print_error("请输入正确的设备型号(pc,board)")
        self.det_model = SCRFD(verbose=verbose,
                               device=device,
                               model_path=det_model_weight_path,
                               target_platform=target_platform
                               )
        self.cls_model = AdaFace(verbose=verbose,
                                 device=device,
                                 model_path=cls_model_weight_path,
                                 target_platform=target_platform
                                 )
        self.database = None

    def create_database(self,
                        picture_folders_path,
                        save_path,
                        thresh=0.5):
        assert os.path.exists(picture_folders_path), "路径{}不存在".format(picture_folders_path)
        assert os.path.exists(save_path), "路径{}不存在".format(save_path)
        labels = os.listdir(picture_folders_path)
        for label in labels:
            picture_folder = os.path.join(picture_folders_path, label)
            face_ls = []
            for pic in os.listdir(picture_folder):
                pic_path = os.path.join(picture_folder, pic)
                if pic_path[-1] == "y":
                    continue
                img = cv2.imread(pic_path)
                bboxes, landmarks = self.det_model.detect(img.copy(), score_thresh=thresh)
                if len(bboxes) == 0:
                    print_standard("{}未检测出人脸".format(pic_path))
                    continue
                else:
                    print_standard("{}检测出人脸，概率为{}".format(pic_path, bboxes[0][-1]))
                img_face = norm_crop(img.copy(), landmarks[0])
                cls_feature = self.cls_model.detect(img_face)[0]
                face_ls.append(cls_feature)
            label_feature = np.stack(face_ls, axis=-1).transpose((1, 0))
            np.save(os.path.join(picture_folder, label + ".npy"), label_feature)

    def export(self, save_path):
        self.cls_model.export(os.path.join(save_path, "cls.rknn"))
        self.det_model.export(os.path.join(save_path, "det.rknn"))

    def load_database(self, database_path):
        assert os.path.exists(database_path), print_error("路径{}不存在".format(database_path))
        labels = os.listdir(database_path)
        database_dic = {}
        for label in labels:
            npy_folder = os.path.join(database_path, label)
            npy_path = os.path.join(npy_folder, label + ".npy")
            assert os.path.exists(npy_path), print_error("路径{}不存在".format(npy_path))
            database_dic[label] = np.load(npy_path)
        self.database = database_dic
        print_standard("数据库中的标签有{}".format(self.database.keys()))
        return database_dic

    def detect(self, img, score_thresh=0.5):
        assert self.database is not None, print_error("还没有读取数据库")
        bboxes, landmarks = self.det_model.detect(img.copy(), score_thresh=score_thresh)
        if len(bboxes) == 0:
            # print("未检测出人脸")
            return None
        feature_ls = []
        for idx, landmark in enumerate(landmarks):
            img_face = norm_crop(img.copy(), landmark)
            cls_feature = self.cls_model.detect(img_face)
            cls_feature = cls_feature.reshape((1, 512))
            feature_dic = {}
            for key in self.database.keys():
                features = self.database[key]
                feature_similarity = 0
                for feature in features:
                    feature = feature.reshape((1, 512))
                    similarity = get_similarity(feature, cls_feature)
                    feature_similarity += similarity
                feature_similarity /= len(features)
                feature_dic[key] = feature_similarity
            feature_dic = sorted(feature_dic.items(), key=lambda x: x[1], reverse=True)
            feature_ls.append(feature_dic.copy())
        return feature_ls, bboxes
