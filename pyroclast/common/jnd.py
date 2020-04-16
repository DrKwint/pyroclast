import tensorflow as tf
import opencv as cv2


def jnd(images):
    """
    Calculate JND mask over each channel of an image

    JND_{\theta}(x, y)=T^{l}(x,y)+T_{\theta}^{t}(x, y)-C_{\theta}^{lt}\cdot\min\{T^{1}(x, y), T_{\theta}^{t}(x, y)\}

    T_l(x,y)  and T_\theta^t(x,y) are the visibility thresholds due to luminance masking and texture masking for a color channel, respectively

    Inputs:
        imgs: NHWC formatted Tensor where C=3 and the image is RGB formatted
    """
    cv.Cvtcolor(CV_RGB2YCrCb)
    print(images.shape)
    patches = tf.image.extract_patches(images,
                                       sizes=[1, 32, 32, 1],
                                       strides=[1, 1, 1, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')  # N, i, j, H*W*C
    bg_luminance
    print(patches.shape)
    dct = tf.signal.dct(patches)  # dct of last dim of N, i, j, H*W*C
    print(dct.shape)
