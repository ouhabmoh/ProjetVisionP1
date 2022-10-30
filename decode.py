import cv2
import numpy as np

getbinary = lambda x, n: format(x, 'b').zfill(n)


def set_bit_to0_at(bin, index):
    return bin & ~(1 << index)


def get_bit_at(bin, index):
    return (bin >> index) & 1


def set_bit_to1_at(bin, index):
    return bin | (1 << index)


def img_to_bin(img, nbbit):
    h, w, c = img.shape
    imgbin = np.zeros(shape=img.shape, dtype=np.uint64)
    for y in range(h):

        for x in range(w):
            for z in range(c):

                imgbin[y, x, z] = getbinary(img[y, x, z], nbbit)

    return imgbin


def set_bit(bin, bit, index):
    if (bit == 1):
        return set_bit_to1_at(bin, index)
    else:
        return set_bit_to0_at(bin, index)


def decode(org_img):
    secret_img = generate_empty_image()
    h, w, c = secret_img.shape
    org_img_cb = org_img[:, :, 1].flatten()
    org_img_cr = org_img[:, :, 2].flatten()
    n = 0
    for y in range(h):
        for x in range(w):
            for z in range(c):
                bs = secret_img[y, x, z]
                bin = ''
                for i in range(2):
                    a = get_bit_at(org_img_cb[n], 4)
                    b = get_bit_at(org_img_cr[n], 4)
                    n += 1
                    c1 = get_bit_at(org_img_cb[n], 5)
                    d = get_bit_at(org_img_cr[n], 5)
                    n += 1
                    secret_img[y,x,z] = set_bit(secret_img[y,x,z], a, i *2)
                    secret_img[y, x, z] = set_bit(secret_img[y, x, z], b, i * 2 + 1)
                    secret_img[y, x, z] = set_bit(secret_img[y, x, z], c1, i * 2 + 2)
                    secret_img[y, x, z] = set_bit(secret_img[y, x, z], d, i * 2 + 3)
                    bin = "{}{}{}{}{}".format(d,c1,b,a,bin)

                bin = int(bin)

                # secret_img[y,x,z] = bin
    return secret_img



def generate_empty_image(h = 80, w = 150):
    return np.ones(shape=(h,w, 3), dtype=np.int8)

def secret_image(text, h=80, w=150):

    sample_img = generate_empty_image(h,w)
    cv2.putText(img=sample_img, text=text, org=(0, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                color=(0, 255, 0), thickness=1)
    return sample_img

img = cv2.imread('image coder.png')

if img is None:
    print('erreur de chargement')
    exit(0)


img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


image_secret_decod = decode(img_y)
cv2.imshow('img', img)

cv2.imshow('decode', image_secret_decod)
cv2.waitKey(0)
cv2.destroyAllWindows()
