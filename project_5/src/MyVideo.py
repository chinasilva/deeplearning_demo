
import cv2
import sys
import os
import random
# import facenet
import numpy as np
# import align.detect_face
import pickle
import joblib
import imageio

def load_kth_data(f_name, data_path, image_size, L):
    """
    :param f_name: video name
    :param data_path: data path
    :param image_size: image size
    :param L: extract L frame of video
    :return: sequence frame of K+T len
    """

    tokens = f_name.split()
    vid_path = os.path.join(data_path, tokens[0] + "_uncomp.avi")
    vid = imageio.get_reader(vid_path, "ffmpeg")  # load video
    low = int(tokens[1])  # start of video
    # make sure the len of video is than L
    high = np.min([int(tokens[2]), vid.get_length()]) - L + 1

    # the len of video is equal L
    if (low == high):
        stidx = 0
    else:
        # the len of video is less-than L, print video path and the error for next line
        if (low >= high): print(vid_path)
        # the len of video greater than L, and the start is random of low-high
        stidx = np.random.randint(low=low, high=high)

    # extract video of L len
    seq = np.zeros((image_size, image_size, L, 1), dtype="float32")
    for t in range(L):
        img = cv2.cvtColor(cv2.resize(vid.get_data(stidx + t), (image_size, image_size)),
                           cv2.COLOR_RGB2GRAY)
        seq[:, :, t] = img[:, :, None]

    return seq

def get_minibatches_idx(n, minibatch_size, shuffle=False):
        """
        :param n: len of data
        :param minibatch_size: minibatch size of data
        :param shuffle: shuffle the data
        :return: len of minibatches and minibatches
        """

        idx_list = np.arange(n, dtype="int32")

        # shuffle
        if shuffle:
            random.shuffle(idx_list)

        # segment
        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                        minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        # processing the last batch
        if (minibatch_start != n):
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

def __getitem__(self, index):

    # read video data of mini-batch with parallel method
    Ls = np.repeat(np.array([self.T + self.K]), self.batch_size, axis=0) # video length of past and feature
    paths = np.repeat(self.root, self.batch_size, axis=0)
    files = np.array(self.trainFiles)[self.mini_batches[index][1]]
    shapes = np.repeat(np.array([self.image_size]), self.batch_size, axis=0)

    with joblib.Parallel(n_jobs=self.batch_size) as parallel:
        output = parallel(joblib.delayed(load_kth_data)(f, p, img_size, l)
                                                            for f, p, img_size, l in zip(files,
                                                                                            paths,
                                                                                            shapes,
                                                                                            Ls))
    # save batch data
    seq_batch = np.zeros((self.batch_size, self.image_size, self.image_size,
                            self.K + self.T, 1), dtype="float32")
    for i in range(self.batch_size):
        seq_batch[i] = output[i]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image

    if self.transform is not None:
        seq_batch = self.transform(seq_batch)


    return seq_batch


def readVideo():
    cap = cv2.VideoCapture('D:\\video\\Wildlife.wmv')
    while(cap.isOpened()):
        ret , frame = cap.read()
        #这里必须加上判断视频是否读取结束的判断,否则播放到最后一帧的时候出现问题了
        if ret == True:
            #gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
            #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #cv2.imshow("frame" , gray)
            cv2.imshow("frame" , frame)
        else:
            break
        #因为视频是10帧每秒，因此每一帧等待100ms
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    readVideo()