import cv2 as cv

if __name__ == '__main__':
    img1 = cv.imread('../Dataset/thumb0.jpg', 0)
    img2 = cv.imread('../Dataset/thumb180.jpg', 0)

    if len(img1.shape) == 3:
        img1 = img1.mean(axis=-1)

    if len(img2.shape) == 3:
        img2 = img2.mean(axis=-1)

    print(img1.shape, img2.shape)

    sift1 = cv.SIFT.create()
    sift2 = cv.SIFT.create()

    keypoints1, des1 = sift1.detectAndCompute(img1, None)
    keypoints2, des2 = sift2.detectAndCompute(img2, None)

    cv.imwrite('../Result/keypoints1_opencv.jpg', cv.drawKeypoints(img1, keypoints1, None))
    cv.imwrite('../Result/keypoints2_opencv.jpg', cv.drawKeypoints(img2, keypoints2, None))

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    img = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good, None, flags=2)
    cv.imwrite('../Result/match_opencv.jpg', img)

