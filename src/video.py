import cv2
import util
import locateFace
import config
import recognize


def openVideo():
    name = input("请输入录入的名称: ")
    cap = cv2.VideoCapture(0)
    while True:
        ok, matLikeImg = cap.read()
        if not ok:
            break
        oneFaceBox = locateFace.locateOneFace(matLikeImg)
        if oneFaceBox is None:
            continue
        drawImg = util.drawFaceBox(config.kUnknownStr, oneFaceBox, matLikeImg)
        cv2.imshow("video", drawImg)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("s"):
            locateFace.locateFaceFromData(matLikeImg, name)
    cap.release()
    cv2.destroyAllWindows()


def videoRecognize():
    cap = cv2.VideoCapture(0)
    idToName, recognizer = recognize.getRecognizer()
    while True:
        ok, matLikeImg = cap.read()
        if not ok:
            break
        ans = recognize.recognizeFromData(matLikeImg, idToName, recognizer)
        for key in ans:
            peopleInfos = ans[key]
            for peopleInfo in peopleInfos:
                peopleName = peopleInfo[0]
                oneFaceBox = peopleInfo[1]
                util.drawFaceBox(peopleName, oneFaceBox, matLikeImg)
            cv2.imshow("video", matLikeImg)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


while True:
    print("1.开启摄像头进行捕获")
    print("2.开启摄像头进行识别")
    print("q.退出")
    choice = input("请选择: ")
    if choice == "1":
        openVideo()
    elif choice == "2":
        videoRecognize()
    else:
        break
