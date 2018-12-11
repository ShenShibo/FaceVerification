import pickle
import cv2
import os
import glob
import numpy as np


data_path = '../webface'
save_path = "../data"
if __name__ == "__main__":
    dirs = os.listdir(data_path)
    label = 0
    labels = []
    resize_shape = 148
    number_dic = {}
    nimg = np.zeros((3, resize_shape, resize_shape), dtype=np.uint8)
    train_data = np.zeros((1, 3, resize_shape,resize_shape), dtype=np.uint8)
    train_label = None
    test_data = np.zeros((1, 3, resize_shape, resize_shape), dtype=np.uint8)
    test_label = None
    for j, dir in enumerate(sorted(dirs)):
        print("name index : {}".format(dir))
        imgs_dir = os.path.join(data_path, dir)
        number = len(os.listdir(imgs_dir))
        number_dic[dir] = number
        for i, img_name in enumerate(os.listdir(imgs_dir)):
            img = cv2.imread(os.path.join(imgs_dir, img_name))
            img = cv2.resize(img, (resize_shape, resize_shape))
            nimg[0, :, :] = img[:, :, 0]
            nimg[1, :, :] = img[:, :, 1]
            nimg[2, :, :] = img[:, :, 2]
            if number <= 5:
                if train_label is None:
                    train_data[0, :, :, :] = nimg
                    train_label = np.array([label], dtype=np.int32)
                else:
                    train_data = np.concatenate((train_data, [nimg]), axis=0)
                    train_label = np.concatenate((train_label, [label]), axis=0)
            elif number <= 14:
                if i < number - 2:
                    if train_label is None:
                        train_data[0, :, :, :] = nimg
                        train_label = np.array([label], dtype=np.int32)
                    else:
                        train_data = np.concatenate((train_data, [nimg]), axis=0)
                        train_label = np.concatenate((train_label, [label]), axis=0)
                else:
                    if test_label is None:
                        test_data[0, :, :, :] = nimg
                        test_label = np.array([label], dtype=np.int32)
                    else:
                        test_data = np.concatenate((test_data, [nimg]), axis=0)
                        test_label = np.concatenate((test_label, [label]), axis=0)
            else:
                if i < number - 5:
                    if train_label is None:
                        train_data[0, :, :, :] = nimg
                        train_label = np.array([label], dtype=np.int32)
                    else:
                        train_data = np.concatenate((train_data, [nimg]), axis=0)
                        train_label = np.concatenate((train_label, [label]), axis=0)
                else:
                    if test_label is None:
                        test_data[0, :, :, :] = nimg
                        test_label = np.array([label], dtype=np.int32)
                    else:
                        test_data = np.concatenate((test_data, [nimg]), axis=0)
                        test_label = np.concatenate((test_label, [label]), axis=0)
        label += 1
        if j > 20:
            break
    with open(os.path.join(save_path, 'webface_train.p'), 'wb') as f:
        d = {'data':train_data, 'labels':train_label}
        pickle.dump(d, f)
    with open(os.path.join(save_path, 'webface_test.p'), 'wb') as f:
        d = {'data':test_data, 'labels':test_label}
        pickle.dump(d, f)


    print(train_label.shape)
    print(train_data.shape)
    print(test_label.shape)
    print(test_data.shape)
            # cv2.imshow('w', img)
            # key = cv2.waitKey()
            # if key == ord('q'):
            #     break