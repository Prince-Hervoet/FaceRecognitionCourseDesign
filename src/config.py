import cv2

# 图片路径
kPhotosPath = "../dataPhotos/"
# 处理过的图片的路径
kSolvedPhotosPath = "../solvedPhotos/"
# 训练数据路径
kTrainDataPath = "../trainData/"
# 测试数据集路径
kTestSetPath = "../testSet/"
# 训练数据文件名
kTrainDataFileName = "train.xml"
# 名称和id映射文件名
kIdToNameFileName = "idToName"
# 脸部识别分类器路径
kFaceCascade = cv2.CascadeClassifier("./classifiers/haarcascade_frontalface_alt2.xml")
