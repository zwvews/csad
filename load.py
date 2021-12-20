import numpy as np
import os



def binary_c_lab(imgs, bin_num=8):
    N_samples = imgs.shape[0]

    d_label = imgs.reshape((N_samples, -1))
    N_pixels = d_label.shape[1]

    d_ext = np.zeros((N_samples, N_pixels * bin_num))

    for n in range(N_samples):
        l = []
        for i in range(N_pixels):
            a = np.zeros(bin_num)
            a[d_label[n, i]] = 1
            l.append(a)

        d_ext[n] = np.concatenate(l)

    return d_ext
def quantize_imgs(imgs, bin_num=8):
    # sub-sample image
    N = imgs.shape[0]
    ss = imgs.reshape((N, 784, 3))
    q = np.amax(ss, axis=1)
    imgs_red = q.reshape((N, 1, 1, 3))

    # quantize colors
    step = 256 // bin_num
    bins = np.array(range(0, 255, step))
    inds = np.digitize(imgs_red, bins) - 1
    imgs_qnt = bins[inds]

    return imgs_qnt, inds

var = '0.020'
res = 8
data_dir = '/media/zwvews/main/Dataset'
filename = 'mnist_10color_jitter_var_{}.npy'.format(var)
filepath = os.path.join(data_dir, filename)

data = np.load(filepath, encoding='latin1', allow_pickle=True).item()
x_train = data['train_image']
y_train = data['train_label']
x_test = data['test_image']
y_test = data['test_label']


__, inds = quantize_imgs(x_train, res)
c_train = binary_c_lab(inds, res)

__, inds = quantize_imgs(x_test, res)
c_test = binary_c_lab(inds, res)
print(1)