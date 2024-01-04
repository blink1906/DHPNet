import argparse
import os
import numpy as np
import joblib
from dataset import get_dataset, img_batch_tensor2numpy
from PIL import Image
import cv2

def samples_extraction(dataset_root, dataset_name, mode, all_bboxes, save_flow_dir, save_frame_dir):
    num_predicted_frame = 1
    # save samples in chunked file
    if dataset_name == "ped2":
        num_samples_each_chunk = 100000
    elif dataset_name == "avenue":
        num_samples_each_chunk = 200000 if mode == "test" else 20000
    elif dataset_name == "shanghaitech":
        num_samples_each_chunk = 300000 if mode == "test" else 100000
    else:
        raise NotImplementedError("dataset name should be one of ped2,avenue or shanghaitech!")

    # frames dataset
    dataset = get_dataset(
        dataset_name=dataset_name,
        dir=os.path.join(dataset_root, dataset_name),
        context_frame_num=3, context_flow_num=3, mode=mode,
        border_mode="predict", all_bboxes=all_bboxes,
        patch_size=32
    )


    if not os.path.exists(save_flow_dir):
        os.mkdir(save_flow_dir)

    if not os.path.exists(save_frame_dir):
        os.mkdir(save_frame_dir)

    global_sample_id = 0
    cnt = 0
    chunk_id = 0  # chunk file id
    chunked_flow_samples = dict(sample_id=[], motion=[], bbox=[], pred_frame=[])
    chunked_frame_samples = dict(frame=[], pred_frame=[])
    save_dir = "/home/xinkai/TCMP-Ped2/flow_bbox"
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(len(dataset)):
        if idx % 1000 == 0:
            print('Extracting foreground in {}-th frame, {} in total'.format(idx + 1, len(dataset)))

        # [num_bboxes,clip_len,C,patch_size, patch_size]
        frame_batch, frameRange, flow_batch, flowRange, _ = dataset.__getitem__(idx)
        frame_batch = img_batch_tensor2numpy(frame_batch)
        chunked_frame_samples["frame"].append(frame_batch)
        chunked_frame_samples["pred_frame"].append(frameRange[-num_predicted_frame:])

        # all the bboxes in current frame
        cur_bboxes = all_bboxes[idx]
        if len(cur_bboxes) > 0:
            flow_batch = img_batch_tensor2numpy(flow_batch)
            # each STC treated as a sample
            for idx_box in range(cur_bboxes.shape[0]):
                chunked_flow_samples["sample_id"].append(global_sample_id)
                clip_len = flow_batch[idx_box].shape[0]

                for n in range(clip_len):
                    flow = flow_batch[idx_box][n]
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    # flow = np.transpose(flow, (1, 2, 0))

                    # 计算光流的大小和方向
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                    # 转换为 HSV 空间
                    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
                    hsv[..., 0] = ang * 180 / np.pi / 2  # 方向
                    hsv[..., 1] = 255  # 饱和度，设为最大值
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 亮度

                    # 转换为 BGR 空间并保存为图像
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    bgr = bgr.astype(np.uint8)
                    img_path = os.path.join(save_dir, f"flow_bbox_{idx_box}_{n}.png")
                    cv2.imwrite(img_path, bgr)
                chunked_flow_samples["motion"].append(flow_batch[idx_box])
                chunked_flow_samples["bbox"].append(cur_bboxes[idx_box])
                chunked_flow_samples["pred_frame"].append(frameRange[-num_predicted_frame:])  # the frame id of last patch
                global_sample_id += 1
                cnt += 1

                if cnt == num_samples_each_chunk:
                    chunked_flow_samples["sample_id"] = np.array(chunked_flow_samples["sample_id"])
                    chunked_flow_samples["motion"] = np.array(chunked_flow_samples["motion"])
                    chunked_flow_samples["bbox"] = np.array(chunked_flow_samples["bbox"])
                    chunked_flow_samples["pred_frame"] = np.array(chunked_flow_samples["pred_frame"])
                    joblib.dump(chunked_flow_samples, os.path.join(save_flow_dir, "chunked_samples_flow_%02d.pkl" % chunk_id))
                    print("Chunk_flow %d file saved!" % chunk_id)

                    chunk_id += 1
                    cnt = 0
                    del chunked_flow_samples
                    chunked_flow_samples = dict(sample_id=[], motion=[], bbox=[], pred_frame=[])


    chunked_frame_samples["frame"] = np.array(chunked_frame_samples["frame"])
    chunked_frame_samples["pred_frame"] = np.array(chunked_frame_samples["pred_frame"])
    joblib.dump(chunked_frame_samples, os.path.join(save_frame_dir, "chunked_samples_frame_%02d.pkl" % chunk_id))
    print("Chunk_frame %d file saved!" % chunk_id)

    # save the remaining samples
    if len(chunked_flow_samples["sample_id"]) != 0:
        chunked_flow_samples["sample_id"] = np.array(chunked_flow_samples["sample_id"])
        chunked_flow_samples["motion"] = np.array(chunked_flow_samples["motion"])
        chunked_flow_samples["bbox"] = np.array(chunked_flow_samples["bbox"])
        chunked_flow_samples["pred_frame"] = np.array(chunked_flow_samples["pred_frame"])
        joblib.dump(chunked_flow_samples, os.path.join(save_flow_dir, "chunked_samples_flow_%02d.pkl" % chunk_id))
        print("Chunk_flow %d file saved!" % chunk_id)

    print('All samples have been saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_root", type=str, default="/home/liuzhian/hdd4T/code/hf2vad", help='project root path')
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--mode", type=str, default="train", help='train or test data')

    args = parser.parse_args()

    all_bboxes = np.load(
        os.path.join("/home/xinkai/data/ped2/ped2", '%s_bboxes_%s.npy' % (args.dataset_name, args.mode)),
        allow_pickle=True
    )
    if args.mode == "train":
        save_flow_dir = os.path.join("/home/xinkai/data/ped2/ped2", "training", "chunked_samples_flow_")
        save_frame_dir = os.path.join("/home/xinkai/data/ped2/ped2", "training", "chunked_samples_frame_")
    else:
        save_flow_dir = os.path.join("/home/xinkai/data/ped2/ped2", "testing", "chunked_samples_flow_")
        save_frame_dir = os.path.join("/home/xinkai/data/ped2/ped2", "testing", "chunked_samples_frame_")

    samples_extraction(
        dataset_root="/home/xinkai/data/ped2/",
        dataset_name=args.dataset_name,
        mode=args.mode,
        all_bboxes=all_bboxes,
        save_flow_dir=save_flow_dir,
        save_frame_dir=save_frame_dir
    )
