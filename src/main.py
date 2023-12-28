import locateFace
import train
import recognize
import cv2


while True:
    print("====================================================")
    print("              请放入数据集后进行选择:")
    print("              1.对数据集进行人脸截取")
    print("              2.使用灰度图片进行训练")
    print("              3.检测测试数据集的人脸")
    print("              4.退出")
    print("====================================================")
    choice = input("              请选择: ")
    if choice == "1":
        locateFace.locateFaceFromFiles()
    elif choice == "2":
        train.trainData()
    elif choice == "3":
        recognize.recognizeFromFiles()
    else:
        break
