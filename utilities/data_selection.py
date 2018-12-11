import os
import cv2
import glob
import shutil

def test_set():
    data_path = '../webface'
    dirs = os.listdir(data_path)
    train_path = '../data/train'
    test_path = '../data/test'
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    dim = 148
    for j, dir in enumerate(sorted(dirs)):
        print("Dictionary index : {}".format(dir))
        imgs_dir = os.path.join(data_path, dir)
        number = len(os.listdir(imgs_dir))
        for i, img_name in enumerate(os.listdir(imgs_dir)):
            src = cv2.imread(os.path.join(imgs_dir, img_name))
            src = cv2.resize(src, (148, 148))
            if number <= 5:
                os.makedirs(os.path.join(train_path, str(j)), exist_ok=True)
                dst = os.path.join(train_path, str(j), img_name)
                cv2.imwrite(dst, src)
            elif number <= 14:
                if i < number - 2:
                    os.makedirs(os.path.join(train_path, str(j)), exist_ok=True)
                    dst = os.path.join(train_path, str(j), img_name)
                    cv2.imwrite(dst, src)
                else:
                    os.makedirs(os.path.join(test_path, str(j)), exist_ok=True)
                    dst = os.path.join(test_path, str(j), img_name)
                    cv2.imwrite(dst, src)
            else:
                if i < number - 5:
                    os.makedirs(os.path.join(train_path, str(j)), exist_ok=True)
                    dst = os.path.join(train_path, str(j), img_name)
                    cv2.imwrite(dst, src)
                else:
                    os.makedirs(os.path.join(test_path, str(j)), exist_ok=True)
                    dst = os.path.join(test_path, str(j), img_name)
                    cv2.imwrite(dst, src)
        if j > 50:
            break

if __name__ == "__main__":
    test_set()

