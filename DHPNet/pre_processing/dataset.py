import torch
import numpy as np
import cv2
from collections import OrderedDict
import os
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import joblib


def get_inputs(file_addr):
    file_format = file_addr.split('.')[-1]
    if file_format == 'mat':
        return sio.loadmat(file_addr, verify_compressed_data_integrity=False)['uv']
    elif file_format == 'npy':
        return np.load(file_addr)
    else:
        return cv2.imread(file_addr)


def img_tensor2numpy(img):
    # mutual transformation between ndarray-like imgs and Tensor-like images
    # both intensity and rgb images are represented by 3-dim data
    if isinstance(img, np.ndarray):
        return torch.from_numpy(np.transpose(img, [2, 0, 1]))
    else:
        return np.transpose(img, [1, 2, 0]).numpy()


def img_batch_tensor2numpy(img_batch):
    # both intensity and rgb image batch are represented by 4-dim data
    if isinstance(img_batch, np.ndarray):
        if len(img_batch.shape) == 4:
            return torch.from_numpy(np.transpose(img_batch, [0, 3, 1, 2]))
        else:
            return torch.from_numpy(np.transpose(img_batch, [0, 1, 4, 2, 3]))
    else:
        if len(img_batch.numpy().shape) == 4:
            return np.transpose(img_batch, [0, 2, 3, 1]).numpy()
        else:
            return np.transpose(img_batch, [0, 1, 3, 4, 2]).numpy()

# 这两段Python代码的功能是实现PyTorch张量(Tensor)和Numpy数组(ndarray)之间的相互转换，并处理其中的维度变换问题。
#
# 1. img_tensor2numpy
# 这个函数负责将单个图像在PyTorch张量（Tensor）和Numpy数组（ndarray）之间进行转换。
#
# 首先，检查输入的img是否为Numpy数组类型（np.ndarray）。如果是，就将其转换为PyTorch张量。这个过程中，对数组进行了转置操作（np.transpose），把原来的[height, width, channel]格式转为[channel, height, width]的格式，因为PyTorch通常使用这种格式来处理图像数据。
#
# 如果输入img不是Numpy数组，那么就假设它是PyTorch张量。此时，将其转换为Numpy数组，并进行维度的转置操作，将[channel, height, width]的格式转为[height, width, channel]的格式，这是因为在Numpy中，我们通常用后者的格式来处理图像数据。
#
# 2. img_batch_tensor2numpy
# 这个函数负责将一批图像（图像批次）在PyTorch张量（Tensor）和Numpy数组（ndarray）之间进行转换。
#
# 首先，检查输入的img_batch是否为Numpy数组类型（np.ndarray）。如果是，根据其维度是否为4进行不同的处理。
#
# 如果是4维（即[batch_size, height, width, channel]），那么就将其转为PyTorch张量，并在转置过程中将其维度转为[batch_size, channel, height, width]。
# 如果不是4维（那就可能是5维，例如在处理视频或者3D图像数据时），那么将其转为PyTorch张量，并在转置过程中将其维度转为[batch_size, time/depth, channel, height, width]。
# 如果输入img_batch不是Numpy数组，那么就假设它是PyTorch张量。此时，也是根据其维度是否为4进行不同的处理，但是转换的目标是Numpy数组，并且转置的方向与前面相反。
#
# 这两个函数的主要目的是帮助我们在处理图像或图像批次时，能够方便地在PyTorch张量和Numpy数组之间进行转换，同时处理好维度的变换问题，以满足两种数据结构在处理图像数据时的习惯。

class bbox_collate:
    def __init__(self, mode):
        self.mode = mode

    def collate(self, batch):
        if self.mode == 'train':
            return bbox_collate_train(batch)
        elif self.mode == 'test':
            return bbox_collate_test(batch)
        else:
            raise NotImplementedError


def bbox_collate_train(batch):
    batch_data = [x[0] for x in batch]
    batch_target = [x[1] for x in batch]
    return torch.cat(batch_data, dim=0), batch_target


def bbox_collate_test(batch):
    batch_data = [x[0] for x in batch]
    batch_target = [x[1] for x in batch]
    return batch_data, batch_target


def get_foreground(img, bboxes, patch_size):
    """
    Cropping the object area according to the bouding box, and resize to patch_size
    :param img: [#frame,c,h,w]
    :param bboxes: [#,4]
    :param patch_size: 32
    :return:
    """
    img_patches = list()
    if len(img.shape) == 3:
        for i in range(len(bboxes)):
            x_min, x_max = int(np.ceil(bboxes[i][0])), int(np.ceil(bboxes[i][2]))
            y_min, y_max = int(np.ceil(bboxes[i][1])), int(np.ceil(bboxes[i][3]))
            cur_patch = img[:, y_min:y_max, x_min:x_max]
            cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]), (patch_size, patch_size))
            img_patches.append(np.transpose(cur_patch, [2, 0, 1]))
        img_patches = np.array(img_patches)
    elif len(img.shape) == 4:
        for i in range(len(bboxes)):
            x_min, x_max = int(np.ceil(bboxes[i][0])), int(np.ceil(bboxes[i][2]))
            y_min, y_max = int(np.ceil(bboxes[i][1])), int(np.ceil(bboxes[i][3]))
            cur_patch_set = img[:, :, y_min:y_max, x_min:x_max]
            tmp_set = list()
            for j in range(img.shape[0]):  # temporal patches
                cur_patch = cur_patch_set[j]
                cur_patch = cv2.resize(np.transpose(cur_patch, [1, 2, 0]),
                                       (patch_size, patch_size))
                tmp_set.append(np.transpose(cur_patch, [2, 0, 1]))
            cur_cube = np.array(tmp_set)  # spatial-temporal cube for each bbox
            img_patches.append(cur_cube)  # all spatial-temporal cubes in a single frame
        img_patches = np.array(img_patches)
    return img_patches  # [num_bboxes,frames_num,C,patch_size, patch_size]


class common_dataset(Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, indice):
        raise NotImplementedError

    def _context_range(self, indice, context_num, tot_num, video_idx):
        """
        get a clip according to the indice (i.e., the frame to be predicted)
        :param indice: be consistent with __getitem__()
        :return: the frame indices in the clip
        """
        if self.border_mode == "predict":
            if indice - context_num < 0:
                start_idx = 0
            else:
                start_idx = indice - context_num
            end_idx = indice
            need_ctx_frames = context_num + 1  # future frame prediction
        else:
            if indice - context_num < 0:
                start_idx = 0
            else:
                start_idx = indice - context_num

            if indice + context_num > tot_num - 1:
                end_idx = tot_num - 1
            else:
                end_idx = indice + context_num
            need_ctx_frames = 2 * context_num + 1

        center_frame_video_idx = video_idx[indice]
        clip_frames_video_idx = video_idx[start_idx:end_idx + 1]
        need_pad = need_ctx_frames - len(clip_frames_video_idx)

        if need_pad > 0:
            if start_idx == 0:
                clip_frames_video_idx = [clip_frames_video_idx[0]] * need_pad + clip_frames_video_idx
            else:
                clip_frames_video_idx = clip_frames_video_idx + [clip_frames_video_idx[-1]] * need_pad

        tmp = np.array(clip_frames_video_idx) - center_frame_video_idx
        offset = np.sum(tmp)

        if tmp[0] != 0 and tmp[-1] != 0:  # extreme condition that is not likely to happen
            print('The video is too short or the context frame number is too large!')
            raise NotImplementedError

        if need_pad == 0 and offset == 0:
            idx = [x for x in range(start_idx, end_idx + 1)]
            return idx
        else:
            if self.border_mode == 'predict':
                if need_pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                idx = [x for x in range(start_idx - offset, end_idx + 1)]
                idx = [idx[0]] * np.maximum(np.abs(offset), need_pad) + idx
                return idx
            else:
                if need_pad > 0 and np.abs(offset) > 0:
                    print('The video is too short or the context frame number is too large!')
                    raise NotImplementedError
                if offset > 0:
                    idx = [x for x in range(start_idx, end_idx - offset + 1)]
                    idx = idx + [idx[-1]] * np.abs(offset)  # 把下一个视频的第一帧换成上一个视频的最后一帧
                    return idx
                elif offset < 0:
                    idx = [x for x in range(start_idx - offset, end_idx + 1)]
                    idx = [idx[0]] * np.abs(offset) + idx
                    return idx
                if need_pad > 0:
                    if start_idx == 0:
                        idx = [x for x in range(start_idx, end_idx + 1)]
                        idx = [idx[0]] * need_pad + idx
                        return idx
                    else:
                        idx = [x for x in range(start_idx, end_idx + 1)]
                        idx = idx + [idx[-1]] * need_pad
                        return idx

class ped_dataset(common_dataset):
    '''
    Loading dataset for UCSD ped2
    '''

    def __init__(self, dir, mode='train', context_frame_num=0, context_flow_num=0, border_mode="hard",
                 frame_format='.jpg', flow_format='.npy', all_bboxes=None, patch_size=32):
        super(ped_dataset, self).__init__()
        self.dir = dir
        self.mode = mode
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.all_flow_addr = list()
        self.frame_video_idx = list()
        self.flow_video_idx = list()
        self.tot_frame_num = 0
        self.tot_flow_num = 0
        self.context_frame_num = context_frame_num
        self.context_flow_num = context_flow_num
        self.border_mode = border_mode
        self.frame_format = frame_format
        self.flow_format = flow_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size


        self.return_gt = False
        if mode == 'test':
            self.all_gt_frame_addr = list()
            self.gts_frame = OrderedDict()
            self.all_gt_flow_addr = list()
            self.gts_flow = OrderedDict()

        self._dataset_init()

    def __len__(self):
        return self.tot_frame_num

    def _dataset_init(self):
        if self.mode == 'train':
            data_frame_dir = os.path.join(self.dir, 'training', 'frames')
            data_flow_dir = os.path.join(self.dir, 'training', "flows")
        elif self.mode == 'test':
            data_frame_dir = os.path.join(self.dir, 'testing', 'frames')
            data_flow_dir = os.path.join(self.dir, 'testing', "flows")
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_frame_dir_list = glob.glob(os.path.join(data_frame_dir, '*'))
            video_flow_dir_list = glob.glob(os.path.join(data_flow_dir, '*'))
            idx = 1
            for video in sorted(video_frame_dir_list):
                video_name = video.split('/')[-1]
                if 'Train' in video_name:
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.frame_format))
                    self.videos[video_name]['frame'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            idx = 1
            for video in sorted(video_flow_dir_list):
                video_name = video.split('/')[-1]
                if 'Train' in video_name:
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['flow'] = glob.glob(os.path.join(video, '*' + self.flow_format))
                    self.videos[video_name]['flow'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['flow'])
                    self.flow_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_flow_addr += cont['flow']
            self.tot_flow_num = len(self.all_flow_addr)

        elif self.mode == 'test':
            dir_frame_list = glob.glob(os.path.join(data_frame_dir, '*'))
            video_frame_dir_list = []
            gt_frame_dir_list = []
            for dir in sorted(dir_frame_list):
                if '_gt' in dir:
                    gt_frame_dir_list.append(dir)
                    self.return_gt = True
                else:
                    name = dir.split('/')[-1]
                    if 'Test' in name:
                        video_frame_dir_list.append(dir)

            # load frames for test
            idx = 1
            for video in sorted(video_frame_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.frame_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)


        dir_flow_list = glob.glob(os.path.join(data_flow_dir, '*'))
        video_flow_dir_list = []
        gt_flow_dir_list = []
        for dir in sorted(dir_flow_list):
            if '_gt' in dir:
                gt_flow_dir_list.append(dir)
                self.return_gt = True
            else:
                name = dir.split('/')[-1]
                if 'Test' in name:
                    video_flow_dir_list.append(dir)

        # load frames for test
        idx = 1
        for video in sorted(video_flow_dir_list):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['flow'] = glob.glob(os.path.join(video, '*' + self.flow_format))
            self.videos[video_name]['flow'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['flow'])
            self.flow_video_idx += [idx] * self.videos[video_name]['length']
            idx += 1

        # merge different frames of different videos into one list
        for _, cont in self.videos.items():
            self.all_flow_addr += cont['flow']
        self.tot_flow_num = len(self.all_flow_addr)

        # load ground truth of frames
        if self.return_gt:
            for gt in sorted(gt_flow_dir_list):
                gt_name = gt.split('/')[-1]
                self.gts_flow[gt_name] = {}
                self.gts_flow[gt_name]['gt_frame'] = glob.glob(os.path.join(gt, '*.bmp'))
                self.gts_flow[gt_name]['gt_frame'].sort()

            # merge different frames of different videos into one list
            for _, cont in self.gts_flow.items():
                self.all_gt_flow_addr += cont['gt_frame']

    def __getitem__(self, indice):
        global gt_flow_batch, gt_frame_batch
        if self.mode == "train":
            # frame indices in a clip
            frame_range = self._context_range(indice=indice, context_num = self.context_frame_num,
                                              tot_num = self.tot_frame_num, video_idx = self.frame_video_idx)
            flow_range = self._context_range(indice=indice, context_num = self.context_flow_num,
                                             tot_num = self.tot_flow_num, video_idx = self.flow_video_idx)
            frame_batch = []
            for idx in frame_range:
                # Get the image
                cur_frame = get_inputs(self.all_frame_addr[idx])

                # Resize the image to 224x224. Note that cv2.resize expects the input shape to be (H, W, C)
                cur_frame = cv2.resize(cur_frame, (224, 224))

                # [h,w,c] -> [c,h,w] BGR
                cur_frame = np.transpose(cur_frame, [2, 0, 1])

                frame_batch.append(cur_frame)
            frame_batch = np.array(frame_batch)

            flow_batch = []
            for idx in flow_range:
                # [h,w,c] -> [c,h,w] BGR
                cur_flow = np.transpose(get_inputs(self.all_flow_addr[idx]), [2, 0, 1])
                flow_batch.append(cur_flow)
            flow_batch = np.array(flow_batch)

            if self.all_bboxes is not None:
                # cropping
                # frame_batch = get_foreground(img=frame_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                flow_batch = get_foreground(img=flow_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
            frame_batch = torch.from_numpy(frame_batch)  # [frames_num,C,patch_size, patch_size]
            flow_batch = torch.from_numpy(flow_batch)   # [num_bboxes,frames_num,C,patch_size, patch_size]

            return frame_batch, frame_range, flow_batch, flow_range, torch.zeros(1)

        elif self.mode == "test":
            frame_range = self._context_range(indice=indice, context_num=self.context_frame_num,
                                              tot_num=self.tot_frame_num, video_idx=self.frame_video_idx)
            flow_range = self._context_range(indice=indice, context_num=self.context_flow_num,
                                             tot_num=self.tot_flow_num, video_idx=self.flow_video_idx)
            frame_batch = []
            for idx in frame_range:
                # Get the image
                cur_frame = get_inputs(self.all_frame_addr[idx])

                # Resize the image to 224x224. Note that cv2.resize expects the input shape to be (H, W, C)
                cur_frame = cv2.resize(cur_frame, (224, 224))

                # [h,w,c] -> [c,h,w] BGR
                cur_frame = np.transpose(cur_frame, [2, 0, 1])

                frame_batch.append(cur_frame)
            frame_batch = np.array(frame_batch)

            flow_batch = []
            for idx in flow_range:
                cur_flow = np.transpose(get_inputs(self.all_flow_addr[idx]), [2, 0, 1])  # [3,h,w]
                flow_batch.append(cur_flow)
            flow_batch = np.array(flow_batch)

            if self.all_bboxes is not None:
                # frame_batch = get_foreground(img=frame_batch, bboxes=self.all_bboxes[indice],
                #                              patch_size=self.patch_size)
                flow_batch = get_foreground(img=flow_batch, bboxes=self.all_bboxes[indice],
                                             patch_size=self.patch_size)
            frame_batch = torch.from_numpy(frame_batch)  # [frames_num,C,patch_size, patch_size]
            flow_batch = torch.from_numpy(flow_batch)   # [num_bboxes,frames_num,C,patch_size, patch_size]

            return frame_batch, frame_range, flow_batch, flow_range, torch.zeros(1)  # to unify the interface
        else:
            raise NotImplementedError



class avenue_dataset(common_dataset):
    def __init__(self, dir, mode='train', context_frame_num=0, context_flow_num=0, border_mode="hard",
                 frame_format='.jpg', flow_format='.npy', all_bboxes=None, patch_size=32):
        super(avenue_dataset, self).__init__()
        self.dir = dir
        self.mode = mode
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.all_flow_addr = list()
        self.frame_video_idx = list()
        self.flow_video_idx = list()
        self.tot_frame_num = 0
        self.tot_flow_num = 0
        self.context_frame_num = context_frame_num
        self.context_flow_num = context_flow_num
        self.border_mode = border_mode
        self.frame_format = frame_format
        self.flow_format = flow_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size


        self.return_gt = False
        if mode == 'test':
            self.all_gt_frame_addr = list()
            self.gts_frame = OrderedDict()
            self.all_gt_flow_addr = list()
            self.gts_flow = OrderedDict()

        self._dataset_init()

    def __len__(self):
        return self.tot_frame_num

    def _dataset_init(self):
        if self.mode == 'train':
            data_frame_dir = os.path.join(self.dir, 'training', 'frames')
            data_flow_dir = os.path.join(self.dir, 'training', "flows")
        elif self.mode == 'test':
            data_frame_dir = os.path.join(self.dir, 'testing', 'frames')
            data_flow_dir = os.path.join(self.dir, 'testing', "flows")
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_frame_dir_list = glob.glob(os.path.join(data_frame_dir, '*'))
            video_flow_dir_list = glob.glob(os.path.join(data_flow_dir, '*'))
            idx = 1
            for video in sorted(video_frame_dir_list):
                video_name = video.split('/')[-1]
                if 'Train' in video_name:
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.frame_format))
                    self.videos[video_name]['frame'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            idx = 1
            for video in sorted(video_flow_dir_list):
                video_name = video.split('/')[-1]
                if 'Train' in video_name:
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['flow'] = glob.glob(os.path.join(video, '*' + self.flow_format))
                    self.videos[video_name]['flow'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['flow'])
                    self.flow_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_flow_addr += cont['flow']
            self.tot_flow_num = len(self.all_flow_addr)

        elif self.mode == 'test':
            dir_frame_list = glob.glob(os.path.join(data_frame_dir, '*'))
            video_frame_dir_list = []
            gt_frame_dir_list = []
            for dir in sorted(dir_frame_list):
                if '_gt' in dir:
                    gt_frame_dir_list.append(dir)
                    self.return_gt = True
                else:
                    name = dir.split('/')[-1]
                    if 'Test' in name:
                        video_frame_dir_list.append(dir)

            # load frames for test
            idx = 1
            for video in sorted(video_frame_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.frame_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)


        dir_flow_list = glob.glob(os.path.join(data_flow_dir, '*'))
        video_flow_dir_list = []
        gt_flow_dir_list = []
        for dir in sorted(dir_flow_list):
            if '_gt' in dir:
                gt_flow_dir_list.append(dir)
                self.return_gt = True
            else:
                name = dir.split('/')[-1]
                if 'Test' in name:
                    video_flow_dir_list.append(dir)

        # load frames for test
        idx = 1
        for video in sorted(video_flow_dir_list):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['flow'] = glob.glob(os.path.join(video, '*' + self.flow_format))
            self.videos[video_name]['flow'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['flow'])
            self.flow_video_idx += [idx] * self.videos[video_name]['length']
            idx += 1

        # merge different frames of different videos into one list
        for _, cont in self.videos.items():
            self.all_flow_addr += cont['flow']
        self.tot_flow_num = len(self.all_flow_addr)

        # load ground truth of frames
        if self.return_gt:
            for gt in sorted(gt_flow_dir_list):
                gt_name = gt.split('/')[-1]
                self.gts_flow[gt_name] = {}
                self.gts_flow[gt_name]['gt_frame'] = glob.glob(os.path.join(gt, '*.bmp'))
                self.gts_flow[gt_name]['gt_frame'].sort()

            # merge different frames of different videos into one list
            for _, cont in self.gts_flow.items():
                self.all_gt_flow_addr += cont['gt_frame']

    def __getitem__(self, indice):
        global gt_flow_batch, gt_frame_batch
        if self.mode == "train":
            # frame indices in a clip
            frame_range = self._context_range(indice=indice, context_num = self.context_frame_num,
                                              tot_num = self.tot_frame_num, video_idx = self.frame_video_idx)
            flow_range = self._context_range(indice=indice, context_num = self.context_flow_num,
                                             tot_num = self.tot_flow_num, video_idx = self.flow_video_idx)
            frame_batch = []
            for idx in frame_range:
                # Get the image
                cur_frame = get_inputs(self.all_frame_addr[idx])

                # Resize the image to 224x224. Note that cv2.resize expects the input shape to be (H, W, C)
                cur_frame = cv2.resize(cur_frame, (224, 224))

                # [h,w,c] -> [c,h,w] BGR
                cur_frame = np.transpose(cur_frame, [2, 0, 1])

                frame_batch.append(cur_frame)
            frame_batch = np.array(frame_batch)

            flow_batch = []
            for idx in flow_range:
                # [h,w,c] -> [c,h,w] BGR
                cur_flow = np.transpose(get_inputs(self.all_flow_addr[idx]), [2, 0, 1])
                flow_batch.append(cur_flow)
            flow_batch = np.array(flow_batch)

            if self.all_bboxes is not None:
                # cropping
                # frame_batch = get_foreground(img=frame_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                flow_batch = get_foreground(img=flow_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
            frame_batch = torch.from_numpy(frame_batch)  # [frames_num,C,patch_size, patch_size]
            flow_batch = torch.from_numpy(flow_batch)   # [num_bboxes,frames_num,C,patch_size, patch_size]

            return frame_batch, frame_range, flow_batch, flow_range, torch.zeros(1)

        elif self.mode == "test":
            frame_range = self._context_range(indice=indice, context_num=self.context_frame_num,
                                              tot_num=self.tot_frame_num, video_idx=self.frame_video_idx)
            flow_range = self._context_range(indice=indice, context_num=self.context_flow_num,
                                             tot_num=self.tot_flow_num, video_idx=self.flow_video_idx)
            frame_batch = []
            for idx in frame_range:
                # Get the image
                cur_frame = get_inputs(self.all_frame_addr[idx])

                # Resize the image to 224x224. Note that cv2.resize expects the input shape to be (H, W, C)
                cur_frame = cv2.resize(cur_frame, (224, 224))

                # [h,w,c] -> [c,h,w] BGR
                cur_frame = np.transpose(cur_frame, [2, 0, 1])

                frame_batch.append(cur_frame)
            frame_batch = np.array(frame_batch)

            flow_batch = []
            for idx in flow_range:
                cur_flow = np.transpose(get_inputs(self.all_flow_addr[idx]), [2, 0, 1])  # [3,h,w]
                flow_batch.append(cur_flow)
            flow_batch = np.array(flow_batch)

            if self.all_bboxes is not None:
                # frame_batch = get_foreground(img=frame_batch, bboxes=self.all_bboxes[indice],
                #                              patch_size=self.patch_size)
                flow_batch = get_foreground(img=flow_batch, bboxes=self.all_bboxes[indice],
                                             patch_size=self.patch_size)
            frame_batch = torch.from_numpy(frame_batch)  # [frames_num,C,patch_size, patch_size]
            flow_batch = torch.from_numpy(flow_batch)   # [num_bboxes,frames_num,C,patch_size, patch_size]

            return frame_batch, frame_range, flow_batch, flow_range, torch.zeros(1)  # to unify the interface
        else:
            raise NotImplementedError





class shanghaitech_dataset(common_dataset):
    def __init__(self, dir, mode='train', context_frame_num=0, context_flow_num=0, border_mode="hard",
                 frame_format='.jpg', flow_format='.npy', all_bboxes=None, patch_size=32):
        super(shanghaitech_dataset, self).__init__()
        self.dir = dir
        self.mode = mode
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.all_flow_addr = list()
        self.frame_video_idx = list()
        self.flow_video_idx = list()
        self.tot_frame_num = 0
        self.tot_flow_num = 0
        self.context_frame_num = context_frame_num
        self.context_flow_num = context_flow_num
        self.border_mode = border_mode
        self.frame_format = frame_format
        self.flow_format = flow_format
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size


        self.return_gt = False
        if mode == 'test':
            self.all_gt_frame_addr = list()
            self.gts_frame = OrderedDict()
            self.all_gt_flow_addr = list()
            self.gts_flow = OrderedDict()

        self._dataset_init()

    def __len__(self):
        return self.tot_frame_num

    def _dataset_init(self):
        if self.mode == 'train':
            data_frame_dir = os.path.join(self.dir, 'training', 'frames')
            data_flow_dir = os.path.join(self.dir, 'training', "flows")
        elif self.mode == 'test':
            data_frame_dir = os.path.join(self.dir, 'testing', 'frames')
            data_flow_dir = os.path.join(self.dir, 'testing', "flows")
        else:
            raise NotImplementedError

        if self.mode == 'train':
            video_frame_dir_list = glob.glob(os.path.join(data_frame_dir, '*'))
            video_flow_dir_list = glob.glob(os.path.join(data_flow_dir, '*'))
            idx = 1
            for video in sorted(video_frame_dir_list):
                video_name = video.split('/')[-1]
                if 'Train' in video_name:
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.frame_format))
                    self.videos[video_name]['frame'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                    self.frame_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)

            idx = 1
            for video in sorted(video_flow_dir_list):
                video_name = video.split('/')[-1]
                if 'Train' in video_name:
                    self.videos[video_name] = {}
                    self.videos[video_name]['path'] = video
                    self.videos[video_name]['flow'] = glob.glob(os.path.join(video, '*' + self.flow_format))
                    self.videos[video_name]['flow'].sort()
                    self.videos[video_name]['length'] = len(self.videos[video_name]['flow'])
                    self.flow_video_idx += [idx] * self.videos[video_name]['length']
                    idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_flow_addr += cont['flow']
            self.tot_flow_num = len(self.all_flow_addr)

        elif self.mode == 'test':
            dir_frame_list = glob.glob(os.path.join(data_frame_dir, '*'))
            video_frame_dir_list = []
            gt_frame_dir_list = []
            for dir in sorted(dir_frame_list):
                if '_gt' in dir:
                    gt_frame_dir_list.append(dir)
                    self.return_gt = True
                else:
                    name = dir.split('/')[-1]
                    if 'Test' in name:
                        video_frame_dir_list.append(dir)

            # load frames for test
            idx = 1
            for video in sorted(video_frame_dir_list):
                video_name = video.split('/')[-1]
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.frame_format))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
                self.frame_video_idx += [idx] * self.videos[video_name]['length']
                idx += 1

            # merge different frames of different videos into one list
            for _, cont in self.videos.items():
                self.all_frame_addr += cont['frame']
            self.tot_frame_num = len(self.all_frame_addr)


        dir_flow_list = glob.glob(os.path.join(data_flow_dir, '*'))
        video_flow_dir_list = []
        gt_flow_dir_list = []
        for dir in sorted(dir_flow_list):
            if '_gt' in dir:
                gt_flow_dir_list.append(dir)
                self.return_gt = True
            else:
                name = dir.split('/')[-1]
                if 'Test' in name:
                    video_flow_dir_list.append(dir)

        # load frames for test
        idx = 1
        for video in sorted(video_flow_dir_list):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['flow'] = glob.glob(os.path.join(video, '*' + self.flow_format))
            self.videos[video_name]['flow'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['flow'])
            self.flow_video_idx += [idx] * self.videos[video_name]['length']
            idx += 1

        # merge different frames of different videos into one list
        for _, cont in self.videos.items():
            self.all_flow_addr += cont['flow']
        self.tot_flow_num = len(self.all_flow_addr)

        # load ground truth of frames
        if self.return_gt:
            for gt in sorted(gt_flow_dir_list):
                gt_name = gt.split('/')[-1]
                self.gts_flow[gt_name] = {}
                self.gts_flow[gt_name]['gt_frame'] = glob.glob(os.path.join(gt, '*.bmp'))
                self.gts_flow[gt_name]['gt_frame'].sort()

            # merge different frames of different videos into one list
            for _, cont in self.gts_flow.items():
                self.all_gt_flow_addr += cont['gt_frame']

    def __getitem__(self, indice):
        global gt_flow_batch, gt_frame_batch
        if self.mode == "train":
            # frame indices in a clip
            frame_range = self._context_range(indice=indice, context_num = self.context_frame_num,
                                              tot_num = self.tot_frame_num, video_idx = self.frame_video_idx)
            flow_range = self._context_range(indice=indice, context_num = self.context_flow_num,
                                             tot_num = self.tot_flow_num, video_idx = self.flow_video_idx)
            frame_batch = []
            for idx in frame_range:
                # Get the image
                cur_frame = get_inputs(self.all_frame_addr[idx])

                # Resize the image to 224x224. Note that cv2.resize expects the input shape to be (H, W, C)
                cur_frame = cv2.resize(cur_frame, (224, 224))

                # [h,w,c] -> [c,h,w] BGR
                cur_frame = np.transpose(cur_frame, [2, 0, 1])

                frame_batch.append(cur_frame)
            frame_batch = np.array(frame_batch)

            flow_batch = []
            for idx in flow_range:
                # [h,w,c] -> [c,h,w] BGR
                cur_flow = np.transpose(get_inputs(self.all_flow_addr[idx]), [2, 0, 1])
                flow_batch.append(cur_flow)
            flow_batch = np.array(flow_batch)

            if self.all_bboxes is not None:
                # cropping
                # frame_batch = get_foreground(img=frame_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
                flow_batch = get_foreground(img=flow_batch, bboxes=self.all_bboxes[indice], patch_size=self.patch_size)
            frame_batch = torch.from_numpy(frame_batch)  # [frames_num,C,patch_size, patch_size]
            flow_batch = torch.from_numpy(flow_batch)   # [num_bboxes,frames_num,C,patch_size, patch_size]

            return frame_batch, frame_range, flow_batch, flow_range, torch.zeros(1)

        elif self.mode == "test":
            frame_range = self._context_range(indice=indice, context_num=self.context_frame_num,
                                              tot_num=self.tot_frame_num, video_idx=self.frame_video_idx)
            flow_range = self._context_range(indice=indice, context_num=self.context_flow_num,
                                             tot_num=self.tot_flow_num, video_idx=self.flow_video_idx)
            frame_batch = []
            for idx in frame_range:
                # Get the image
                cur_frame = get_inputs(self.all_frame_addr[idx])

                # Resize the image to 224x224. Note that cv2.resize expects the input shape to be (H, W, C)
                cur_frame = cv2.resize(cur_frame, (224, 224))

                # [h,w,c] -> [c,h,w] BGR
                cur_frame = np.transpose(cur_frame, [2, 0, 1])

                frame_batch.append(cur_frame)
            frame_batch = np.array(frame_batch)

            flow_batch = []
            for idx in flow_range:
                cur_flow = np.transpose(get_inputs(self.all_flow_addr[idx]), [2, 0, 1])  # [3,h,w]
                flow_batch.append(cur_flow)
            flow_batch = np.array(flow_batch)

            if self.all_bboxes is not None:
                # frame_batch = get_foreground(img=frame_batch, bboxes=self.all_bboxes[indice],
                #                              patch_size=self.patch_size)
                flow_batch = get_foreground(img=flow_batch, bboxes=self.all_bboxes[indice],
                                             patch_size=self.patch_size)
            frame_batch = torch.from_numpy(frame_batch)  # [frames_num,C,patch_size, patch_size]
            flow_batch = torch.from_numpy(flow_batch)   # [num_bboxes,frames_num,C,patch_size, patch_size]

            return frame_batch, frame_range, flow_batch, flow_range, torch.zeros(1)  # to unify the interface
        else:
            raise NotImplementedError



def get_dataset(dataset_name, dir, mode='train', context_frame_num=0, context_flow_num=0, border_mode='hard',
                all_bboxes=None, patch_size=32):

    frame_ext = {"ped2": ".jpg", "avenue": ".jpg", "shanghaitech": ".jpg"}[dataset_name]
    flow_ext = ".npy"

    if dataset_name == "ped2":
        dataset = ped_dataset(dir=dir, context_frame_num=context_frame_num, context_flow_num=context_flow_num, mode=mode, border_mode=border_mode,
                              all_bboxes=all_bboxes, patch_size=patch_size, frame_format=frame_ext,
                              flow_format=flow_ext)
    elif dataset_name == 'avenue':
        dataset = avenue_dataset(dir=dir, context_frame_num=context_frame_num, context_flow_num=context_flow_num, mode=mode, border_mode=border_mode,
                                 all_bboxes=all_bboxes, patch_size=patch_size, frame_format=frame_ext, flow_format=flow_ext)
    elif dataset_name == 'shanghaitech':
        dataset = shanghaitech_dataset(dir=dir, context_frame_num=context_frame_num, context_flow_num=context_flow_num,mode=mode, border_mode=border_mode,
                                       all_bboxes=all_bboxes, patch_size=patch_size, frame_format=frame_ext,flow_format=flow_ext)
    else:
        raise NotImplementedError

    return dataset


transform = transforms.Compose([
    transforms.ToTensor(),
])

transform_frame = transforms.Compose([
    transforms.Resize((224, 224)),
])


class Chunked_flow_sample_dataset(Dataset):
    def __init__(self, chunk_file, last_flow=False, transform=transform):
        super(Chunked_flow_sample_dataset, self).__init__()
        self.chunk_file = chunk_file

        # dict(sample_id=[], motion=[], bbox=[], pred_frame=[])
        self.chunked_samples = joblib.load(self.chunk_file)
        self.chunked_samples_motion = self.chunked_samples["motion"]
        self.chunked_samples_bbox = self.chunked_samples["bbox"]
        self.chunked_samples_pred_frame = self.chunked_samples["pred_frame"]
        self.chunked_samples_id = self.chunked_samples["sample_id"]

        self.transform = transform

    def __len__(self):
        return len(self.chunked_samples_id)

    def __getitem__(self, indice):
        motion = self.chunked_samples_motion[indice]
        bbox = self.chunked_samples_bbox[indice]
        pred_frame = self.chunked_samples_pred_frame[indice]

        y = np.transpose(motion, [1, 2, 0, 3])
        y = np.reshape(y, (y.shape[0], y.shape[1], -1))
        y1 = y[:, :, 0:2]  # flow 1
        y2 = y[:, :, 2:4]  # flow 2
        y3 = y[:, :, 4:6]  # flow 3

        # z = np.transpose(frame, [1, 2, 0, 3])
        # z = np.reshape(z, (z.shape[0], z.shape[1], -1))


        return (self.transform(y1), self.transform(y2), self.transform(y3)),\
               bbox.astype(np.float32), pred_frame, indice

class Chunked_frame_sample_dataset(Dataset):
    def __init__(self, chunk_file, last_flow=False, transform=transform):
        super(Chunked_frame_sample_dataset, self).__init__()
        self.chunk_file = chunk_file

        # dict(sample_id=[], motion=[], bbox=[], pred_frame=[])
        self.chunked_samples = joblib.load(self.chunk_file)
        self.chunked_samples_frame = self.chunked_samples["frame"]
        self.chunked_samples_pred_frame = self.chunked_samples["pred_frame"]

        self.transform = transform

    def __len__(self):
        return len(self.chunked_samples_frame)

    def __getitem__(self, indice):
        frame = self.chunked_samples_frame[indice]
        pred_frame = self.chunked_samples_pred_frame[indice]

        x = np.transpose(frame, [1, 2, 0, 3])
        x = np.reshape(x, (x.shape[0], x.shape[1], -1))

        return self.transform(x), pred_frame, indice