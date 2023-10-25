import numpy as np
import cv2 as cv


def gaussian_kernel(sigma, dim):
    kernel = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            kernel[i, j] = np.exp(-((i - dim // 2) ** 2 + (j - dim // 2) ** 2) / (2 * sigma ** 2))
    return kernel / (2 * np.pi * sigma ** 2)


def convolve(img, kernel, pad):
    mat = np.pad(img, ((pad, pad), (pad, pad)), 'constant')
    img_conv = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_conv[i, j] = np.multiply(kernel, mat[i:i + kernel.shape[0],
                                                 j:j + kernel.shape[1]]).sum()
    return img_conv


def gaussian_blur(img, sigma, dim):
    kernel = gaussian_kernel(sigma, dim)
    img_blur = convolve(img, kernel, dim // 2)
    return img_blur


def downsample(img):
    img_down = img[::2, ::2]
    return img_down


def gaussian_pyramid(img, sigma0, O, S):
    gp = []
    k = np.power(2.0, 1 / S)
    n = S + 3
    for o in range(O):
        ls = []
        for s in range(n):
            sigma = sigma0 * np.power(k, s) * np.power(2, o)
            dim = int(6 * sigma + 1)
            if dim % 2 == 0:
                dim += 1
            ls.append(gaussian_blur(img, sigma, dim))
        img = downsample(img)
        gp.append(np.array(ls))
    return gp


def dog_pyramid(gp, O, S):
    n = S + 2
    dp = []
    for o in range(O):
        ls = []
        for s in range(n):
            ls.append(gp[o][s + 1] - gp[o][s])
        dp.append(np.array(ls))
    return dp


def ck_extrema(mat, val):
    s, w, h = mat.shape
    if val > 0:
        for i in range(s):
            for j in range(w):
                for k in range(h):
                    if i == 1 and j == 1 and k == 1:
                        continue
                    if mat[i, j, k] >= val:
                        return False
    else:
        for i in range(s):
            for j in range(w):
                for k in range(h):
                    if i == 1 and j == 1 and k == 1:
                        continue
                    if mat[i, j, k] <= val:
                        return False
    return True


def find_extrema(dp, O, S, threshold):
    extrema = []
    for o in range(O):
        mat = dp[o]
        for s in range(1, S + 1):
            h, w = mat[s].shape
            for i in range(1, w - 1):
                for j in range(1, h - 1):
                    if (np.abs(mat[s][i, j]) > threshold / S
                            and ck_extrema(mat[s - 1:s + 2, i - 1:i + 2, j - 1:j + 2],
                                           mat[s][i, j])):
                        extrema.append([o, s, i, j])
    return extrema


def derivation(mat, h):
    dx = (mat[1, 1, 2] - mat[1, 1, 0]) * h / 2
    dy = (mat[1, 2, 1] - mat[1, 0, 1]) * h / 2
    ds = (mat[2, 1, 1] - mat[0, 1, 1]) * h / 2
    dD = np.array([dx, dy, ds])
    dxx = (mat[1, 1, 2] + mat[1, 1, 0] - 2 * mat[1, 1, 1]) * h
    dyy = (mat[1, 2, 1] + mat[1, 0, 1] - 2 * mat[1, 1, 1]) * h
    dss = (mat[2, 1, 1] + mat[0, 1, 1] - 2 * mat[1, 1, 1]) * h
    dxy = (mat[1, 2, 2] + mat[1, 0, 0] - mat[1, 2, 0] - mat[1, 0, 2]) * h / 4
    dxs = (mat[2, 1, 2] + mat[0, 1, 0] - mat[2, 1, 0] - mat[0, 1, 2]) * h / 4
    dys = (mat[2, 2, 1] + mat[0, 0, 1] - mat[2, 0, 1] - mat[0, 2, 1]) * h / 4
    dH = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
    return dD, dH


def adjust_fit(d, o, s, x, y, S, threshold):
    scale = 1.0 / 255

    flag = False
    mat, dD, dH, X = None, None, None, None
    for T in range(5):
        mat = d[s - 1:s + 2, x - 1:x + 2, y - 1:y + 2]
        dD, dH = derivation(mat, scale)
        X = -np.matmul(np.linalg.pinv(dH), dD)
        if np.abs(X[0]) < 0.5 and np.abs(X[1]) < 0.5 and np.abs(X[2]) < 0.5:
            flag = True
            break
        x += int(np.round(X[1]))
        y += int(np.round(X[0]))
        s += int(np.round(X[2]))
        if s < 1 or s > S or x < 5 or x > d[s].shape[0] - 5 or y < 5 or y > d[s].shape[1] - 5:
            break
    if flag:
        D = np.dot(dD, X)
        if np.abs(d[s, x, y] * scale + 0.5 * D) < threshold / S:
            return None
        return o, s, x, y
    return None


def keypoints_fit(extrema, dp, O, S, threshold):
    keypoints = []

    for (o, s, x, y) in extrema:
        keypoint = adjust_fit(dp[o], o, s, x, y, S, threshold)
        if keypoint is not None:
            keypoints.append(keypoint)

    return keypoints


def eliminating_edge(mat, o, s, x, y, threshold):
    scale = 1.0 / 255

    dD, dH = derivation(mat, scale)
    H = dH[0:2, 0:2]
    tr = np.trace(H)
    det = np.linalg.det(H)
    if det <= 0 or tr ** 2 / det > (threshold + 1) ** 2 / threshold:
        return None
    return o, s, x, y


def keypoints_edge(keypoints, dp, O, S, threshold):
    keypoints_new = []

    for (o, s, x, y) in keypoints:
        keypoint = eliminating_edge(dp[o][s - 1:s + 2, x - 1:x + 2, y - 1:y + 2], o, s, x, y, threshold)
        if keypoint is not None:
            keypoints_new.append(keypoint)

    return keypoints_new


def direentation_assignment(mat, o, s, x, y, sigma):
    sigma *= 1.5

    r = int(np.round(3 * sigma))

    dire = []
    w = []

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            now_x = x + i
            now_y = y + j
            if now_x < 1 or now_x > mat.shape[0] - 2 or now_y < 1 or now_y > mat.shape[1] - 2:
                continue
            dx = mat[now_x, now_y + 1] - mat[now_x, now_y - 1]
            dy = mat[now_x + 1, now_y] - mat[now_x - 1, now_y]
            dire.append(np.arctan2(dy, dx) * 180 / np.pi)
            w.append(np.sqrt(dx ** 2 + dy ** 2) * np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2)))

    sum = np.zeros(36)

    for i in range(len(w)):
        p = int(np.round(dire[i] / 10))
        if p < 0:
            p += 36
        if p >= 36:
            p -= 36
        sum[p] += w[i]

    main_dire = np.argmax(sum)
    final_dire = []

    for i in range(len(sum)):
        if sum[i] >= 0.8 * sum[main_dire]:
            final_dire.append(i)

    return final_dire


def keypoints_dire(keypoints, gp, O, S, sigma0):
    keypoints_new = []

    for (o, s, x, y) in keypoints:
        sigma = sigma0 * np.power(2.0, s / S)
        keypoint = direentation_assignment(gp[o][s], o, s, x, y, sigma)
        for dire in keypoint:
            keypoints_new.append([o, s, x, y, dire])

    return keypoints_new


def func(x, y):
    if x == 0:
        return 1 - y
    return y


def descriptor(mat, o, s, x, y, dire, sigma, d, n):
    cos_t, sin_t = np.cos(dire * 10 * np.pi / 180) / (3 * sigma), np.sin(dire * 10 * np.pi / 180) / (3 * sigma)
    r = int(np.round(3 * sigma * np.sqrt(2) * (d + 1) * 0.5))

    Bin_x = []
    Bin_y = []
    w = []
    Bin_o = []

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            rot_x = i * cos_t + j * sin_t
            rot_y = -i * sin_t + j * cos_t
            bin_x = rot_x + d // 2 - 0.5
            bin_y = rot_y + d // 2 - 0.5
            if bin_x <= -1 or bin_x >= d or bin_y <= -1 or bin_y >= d:
                continue
            now_x = x + i
            now_y = y + j
            if now_x < 1 or now_x > mat.shape[0] - 2 or now_y < 1 or now_y > mat.shape[1] - 2:
                continue
            dx = mat[now_x, now_y + 1] - mat[now_x, now_y - 1]
            dy = mat[now_x + 1, now_y] - mat[now_x - 1, now_y]

            Bin_x.append(bin_x)
            Bin_y.append(bin_y)
            w.append(np.sqrt(dx ** 2 + dy ** 2) * np.exp(-(rot_x ** 2 + rot_y ** 2) / (0.5 * d ** 2)))
            Bin_o.append((np.arctan2(dy, dx) * 180 / np.pi - dire * 10) * n / 360)

    W = np.zeros((d, d, n))

    for k in range(len(w)):
        bin_x = Bin_x[k]
        bin_y = Bin_y[k]
        bin_o = Bin_o[k]

        x0 = int(bin_x)
        y0 = int(bin_y)
        o0 = int(bin_o)
        bin_x -= x0
        bin_y -= y0
        bin_o -= o0

        if o0 < 0:
            o0 += n
        if o0 >= n:
            o0 -= n

        for a in range(2):
            for b in range(2):
                for c in range(2):
                    if 0 <= x0 + a < d and 0 <= y0 + b < d and 0 <= o0 + c < n:
                        W[x0 + a, y0 + b, o0 + c] += w[k] * func(a, bin_x) * func(b, bin_y) * func(c, bin_o)

    W = W.reshape(-1)
    W /= np.sqrt(np.sum(np.power(W, 2)) + 1e-7)
    W[W > 0.2] = 0.2
    W /= np.sqrt(np.sum(np.power(W, 2)) + 1e-7)

    return W


def keypoints_describe(keypoints, gp, O, S, sigma0, d, n):
    descriptors = []

    for (o, s, x, y, dire) in keypoints:
        sigma = sigma0 * np.power(2.0, s / S)
        descriptors.append(descriptor(gp[o][s], o, s, x, y, dire, sigma, d, n))

    return descriptors


def SIFT(img):
    sigma0 = 1.6
    sigma_t = 0.5
    t = 3
    O = int(np.log2(min(img.shape[0], img.shape[1])) - t)
    S = 3
    threshold = 0.04
    threshold_d = 10
    d = 4
    n = 8

    gp = gaussian_pyramid(img, np.sqrt(sigma0 ** 2 - sigma_t ** 2), O, S)
    dp = dog_pyramid(gp, O, S)
    extrema = find_extrema(dp, O, S, threshold * 0.5)
    keypoints = keypoints_fit(extrema, dp, O, S, threshold)
    keypoints = keypoints_edge(keypoints, dp, O, S, threshold_d)
    keypoints = keypoints_dire(keypoints, gp, O, S, sigma0)
    descriptors = keypoints_describe(keypoints, gp, O, S, sigma0, d, n)

    print(len(keypoints))

    return keypoints, descriptors


def match(des1, des2):
    threshold = 0.8

    match = []

    for i in range(len(des1)):
        min_dis = 1e10
        min_j = -1
        for j in range(len(des2)):
            dis = np.sqrt(np.sum(np.power(des1[i] - des2[j], 2)))
            if dis < min_dis:
                min_dis = dis
                min_j = j
        match.append([i, min_j])

    return match


def draw_match(img1, img2, keypoints1, keypoints2, match):
    img = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    img[:img1.shape[0], :img1.shape[1]] = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img[:img2.shape[0], img1.shape[1]:] = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for (i, j) in match:
        x1 = int(np.round(keypoints1[i][2]))
        y1 = int(np.round(keypoints1[i][3]))
        x2 = int(np.round(keypoints2[j][2]))
        y2 = int(np.round(keypoints2[j][3] + img1.shape[1]))
        cv.line(img, (y1, x1), (y2, x2), (0, 0, 255), 1)

    return img


def draw_keypoints(img, keypoints):
    for (o, s, x, y, dire) in keypoints:
        cv.circle(img, (int(np.round(y)), int(np.round(x))), 1, (0, 0, 255), 1)
        cv.line(img, (int(np.round(y)), int(np.round(x))),
                (int(np.round(y + 5 * np.cos(dire * 10 * np.pi / 180))),
                 int(np.round(x + 5 * np.sin(dire * 10 * np.pi / 180)))), (0, 0, 255), 1)
    return img


if __name__ == '__main__':
    img1 = cv.imread('../Dataset/thumb0.jpg', 0)
    img2 = cv.imread('../Dataset/thumb180.jpg', 0)

    if len(img1.shape) == 3:
        img1 = img1.mean(axis=-1)

    if len(img2.shape) == 3:
        img2 = img2.mean(axis=-1)

    print(img1.shape, img2.shape)

    keypoints1, descriptors1 = SIFT(img1)
    keypoints2, descriptors2 = SIFT(img2)

    match = match(descriptors1, descriptors2)

    img = draw_match(img1, img2, keypoints1, keypoints2, match)
    cv.imwrite('../Result/match_myself.jpg', img)

    img = draw_keypoints(img1, keypoints1)
    cv.imwrite('../Result/keypoints1_myself.jpg', img)

    img = draw_keypoints(img2, keypoints2)
    cv.imwrite('../Result/keypoints2_myself.jpg', img)
