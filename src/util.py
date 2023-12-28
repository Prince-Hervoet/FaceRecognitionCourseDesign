import shutil
import os
import uuid
import cv2


# 清空文件夹
def clearOldPhotos(pathStr):
    shutil.rmtree(pathStr)
    os.mkdir(pathStr)


# 获取某个目录的子目录名称
def getPhotoDirName(pathStr):
    children = os.listdir(pathStr)
    ans = []
    for file in children:
        if os.path.isfile(os.path.join(pathStr, file)):
            continue
        ans.append(file)
    return ans


# 生成一个随机id
def getUUIDStr():
    return str(uuid.uuid1())


# 绘制人脸框，返回一个副本图片
# name 人名
# faceBox 人脸位置信息
# sourceMatLikeImg 原始Matlike类型的图片
def drawFaceBox(name, faceBox, sourceMatLikeImg):
    ectypeImg = sourceMatLikeImg.copy()
    for x, y, w, h in faceBox:
        cv2.rectangle(ectypeImg, (x, y), (x + w, y + h), color=(0, 0, 255))
        font = cv2.FONT_HERSHEY_SIMPLEX
        textSize = cv2.getTextSize(name, font, 1, 2)
        cv2.rectangle(
            ectypeImg,
            (x, y - textSize[0][1] - textSize[1] - 1),
            (x + textSize[0][0], y),
            (0, 0, 255),
            -1,
        )
        cv2.putText(ectypeImg, name, (x, y - textSize[1]), font, 1, (255, 255, 255), 2)
    return ectypeImg


# 获取最大的人脸
def getMaxFaceBox(faceBox):
    if len(faceBox) < 2:
        return faceBox
    maxArea = 0
    ansIndex = 0
    for index, (x, y, w, h) in enumerate(faceBox):
        area = w * h
        if area > maxArea:
            maxArea = area
            ansIndex = index
    return [faceBox[ansIndex]]
