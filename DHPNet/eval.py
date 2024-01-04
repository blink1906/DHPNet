import argparse
import os
import torch
import cv2
import joblib
import pickle
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
from model.network import flow_net
from model.network_frame import frame_net
from PIL import Image
import math
from pre_processing.dataset import Chunked_flow_sample_dataset, Chunked_frame_sample_dataset
from utils.eval_utils import save_evaluation_curves, save_flow_evaluation_curves
import scipy.signal as signal

METADATA = {
    "ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                               180, 180]
    },
    "avenue": {
        "testing_video_num": 21,
        "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
                               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,
                               76],
    },
    "shanghaitech": {
        "testing_video_num": 107,
        "testing_frames_cnt": [265, 433, 337, 601, 505, 409, 457, 313, 409, 337,
                               337, 457, 577, 313, 529, 193, 289, 289, 265, 241,
                               337, 289, 265, 217, 433, 409, 529, 313, 217, 241,
                               313, 193, 265, 317, 457, 337, 361, 529, 409, 313,
                               385, 457, 481, 457, 433, 385, 241, 553, 937, 865,
                               505, 313, 361, 361, 529, 337, 433, 481, 649, 649,
                               409, 337, 769, 433, 241, 217, 265, 265, 217, 265,
                               409, 385, 481, 457, 313, 601, 241, 481, 313, 337,
                               457, 217, 241, 289, 337, 313, 337, 265, 265, 337,
                               361, 433, 241, 433, 601, 505, 337, 601, 265, 313,
                               241, 289, 361, 385, 217, 337, 265]
    },
    "street":{
        "testing_video_num":9,
        "testing_frames_cnt":[4219, 5200, 4860, 3634, 5200, 5200, 3993, 5200, 5200]
    }

}

def psnr(mse):
    return 10 * math.log10(1 / mse)

def filter(data, template, radius=5):
    arr=np.array(data)
    length=arr.shape[0]
    newData=np.zeros(length)

    for j in range(radius//2,arr.shape[0]-radius//2):
        t=arr[ j-radius//2:j+radius//2+1]
        a=np.multiply(t,template)
        newData[j]=a.sum()
    # expand
    for i in range(radius//2):
        newData[i]=newData[radius//2]
    for i in range(-radius//2,0):
        newData[i]=newData[-radius//2]
    return newData


import os



def calc(r=5, sigma=2):
    k = np.zeros(r)
    for i in range(r):
        k[i] = 1/((2*math.pi)**0.5*sigma)*math.exp(-((i-r//2)**2/2/(sigma**2)))
    return k

def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))

    return anomaly_score_list

def evaluate(config, ckpt_path_flow, ckpt_path_frame, testing_chunked_flow_samples_file, testing_chunked_frame_samples_file, training_stats_path_flow, suffix):
    global flow_mean, flow_std, frame_mean, frame_std, correspondence_mean, correspondence_std
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    device = config["device"]
    num_workers = config["num_workers"]

    testset_num_frames = np.sum(METADATA[dataset_name]["testing_frames_cnt"])

    eval_dir = os.path.join(config["eval_root"], config["exp_name"])
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    model_flow = flow_net().to(device).eval()
    model_flow_weights = torch.load(ckpt_path_flow)["model_state_dict"]
    model_flow.load_state_dict(model_flow_weights)
    print("load pre-trained success!")

    #  get training stats
    if training_stats_path_flow is not None:
        training_scores_stats = torch.load(training_stats_path_flow)

        flow_mean, flow_std = np.mean(training_scores_stats["flow_training_stats"]), \
                          np.std(training_scores_stats["flow_training_stats"])

    score_func = nn.MSELoss(reduction="none")

    dataset_test_flow = Chunked_flow_sample_dataset(testing_chunked_flow_samples_file)
    dataloader_test_flow = DataLoader(dataset=dataset_test_flow, batch_size=128, num_workers=num_workers, shuffle=False)

    model_frame = frame_net(proto_size=config["model_paras"]["proto_size"],
                 feature_dim=config["model_paras"]["feature_dim"],
                 proto_dim=config["model_paras"]["proto_dim"],
                 shink_thres=config["model_paras"]["shrink_thres"],
                 dropout=config["model_paras"]["drop_out"]).to(device).eval()

    model_frame_weights = torch.load(ckpt_path_frame)["model_state_dict"]
    model_frame.load_state_dict(model_frame_weights)



    score_func = nn.MSELoss(reduction="none")

    dataset_test_frame = Chunked_frame_sample_dataset(testing_chunked_frame_samples_file)
    dataloader_test_frame = DataLoader(dataset=dataset_test_frame, batch_size=1, num_workers=num_workers, shuffle=False)

    flow_bbox_scores = [{} for i in range(testset_num_frames.item())]
    for test_data in tqdm(dataloader_test_flow, desc="Eval: ", total=len(dataloader_test_flow)):

        flow_tuple_test, bbox_test, pred_frame_test, indices_test = test_data
        flow_tuple_test = [flow.to(device) for flow in flow_tuple_test]

        out_test = model_flow(flow_tuple_test, mode="test")

        temp_loss_flow = None
        mse_loss_flow = None
        save_dir = "/home/xinkai/TCMP-ped2/error_map_flow_ped2/"
        for output_flow, target_flow in zip(out_test["output_flow_tuple"], flow_tuple_test):

            loss_flow = score_func(output_flow, target_flow).cpu().data.numpy()
            loss_mse = score_func((output_flow+1)/2, (target_flow+1)/2).cpu().data.numpy()
            if temp_loss_flow is None:
                temp_loss_flow = loss_flow
            else:
                temp_loss_flow += loss_flow

            if mse_loss_flow is None:
                mse_loss_flow = loss_mse
            else:
                mse_loss_flow += loss_mse


        loss_flow = temp_loss_flow / len(out_test["output_flow_tuple"])
        loss_mse = mse_loss_flow / len(out_test["output_flow_tuple"])

        flow_scores = np.sum(np.sum(np.sum(loss_flow, axis=3), axis=2), axis=1)
        mse_scores = np.sum(np.sum(np.sum(loss_mse, axis=3), axis=2), axis=1)
        if training_stats_path_flow is not None:
            flow_scores = (flow_scores - flow_mean) / flow_std
            mse_scores = (mse_scores - flow_mean) / flow_std

        for i in range(len(flow_scores)):
            flow_bbox_scores[pred_frame_test[i][-1].item()][i] = flow_scores[i]

    del dataset_test_flow

    frame_scores = []
    flow_scores = np.empty(len(flow_bbox_scores))
    flow_scores_max = np.empty(len(flow_bbox_scores))

    cnt = 0
    for test_data in tqdm(dataloader_test_frame, desc="Eval: ", total=len(dataloader_test_frame)):
        frame, pred_frame_test, indices_test = test_data
        frame = frame.to(device)

        out = model_frame(frame, mode="test")
        out["output_frame"] = out["output_frame"].to(device)
        out["frame_target"] = out["frame_target"].to(device)

        mse_frame_ = score_func((out["output_frame"] + 1) / 2, (out["frame_target"] + 1) / 2)

        mse_frame = mse_frame_.view((mse_frame_.shape[0], -1))
        mse_frame = mse_frame.mean(-1)
        for j in range(len(mse_frame)):
            psnr_score = psnr(mse_frame[j].item())
            frame_scores.append(psnr_score)
        cnt += 1


    del dataset_test_frame
    frame_scores = np.array(frame_scores)

    for i in range(len(flow_scores)):
        flow_scores[i] = np.max(list(flow_bbox_scores[i].values()))
        flow_scores_max[i] = np.sum(list(flow_bbox_scores[i].values()))



    # ================== Calculate AUC ==============================
    # load gt labels
    gt = pickle.load(
        open(os.path.join(config["dataset_base_dir"], "%s/ground_truth_demo/gt_label.json" % dataset_name), "rb"))
    gt_concat = np.concatenate(list(gt.values()), axis=0)

    new_gt = np.array([])
    new_frame_scores = np.array([])
    new_flow_scores = np.array([])

    start_idx = 0
    for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
        gt_each_video = gt_concat[start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]
        frame_scores_each_video = frame_scores[
                                  start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]
        flow_scores_each_video_max = flow_scores_max[
                            start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]
        template = calc(15, 2)
        frame_scores_ = anomaly_score_list(frame_scores_each_video)
        frame_scores_each_video = filter(frame_scores_, template, 15)
        scores = anomaly_score_list(flow_scores_each_video_max)
        flow_scores_each_video_max = filter(scores, template, 15)
        flow_scores_each_video = flow_scores[
                                  start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]
        flow_scores_each_video = flow_scores_each_video + 40 * flow_scores_each_video_max
        frame_scores_each_video = 1 - frame_scores_each_video

        start_idx += METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]

        new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
        new_frame_scores = np.concatenate((new_frame_scores, frame_scores_each_video), axis=0)
        new_flow_scores = np.concatenate((new_flow_scores, flow_scores_each_video), axis=0)

    gt_concat = new_gt
    frame_scores = new_frame_scores
    flow_scores = new_flow_scores


    curves_save_path = os.path.join(config["eval_root"], config["exp_name"], 'anomaly_curves_%s' % suffix)
    auc, aupr, eer = save_evaluation_curves(frame_scores, flow_scores, gt_concat, curves_save_path,
                                 np.array(METADATA[dataset_name]["testing_frames_cnt"]) - 4)

    return auc, aupr, eer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--flowmodel_save_path", type=str,
                        default="/home/xinkai/TCMP-ped2/ckpt_flow/model.pth-2",
                        help='path to pretrained weights')
    parser.add_argument("--framemodel_save_path", type=str,
                        default="/home/xinkai/TCMP-ped2/ckpt_frame_ped2/ped2_TCMP/model.pth-10",
                        help='path to pretrained weights')
    parser.add_argument("--cfg_file", type=str,
                        default="/home/xinkai/TCMP-ped2/TCMP-main/TCMP-main/cfgs/cfg.yaml",
                        help='path to pretrained model configs')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.cfg_file))
    testing_chunked_flow_samples_file = "/home/xinkai/data/ped2/ped2/testing/chunked_samples_flow/chunked_samples_flow_00.pkl"
    testing_chunked_frame_samples_file = "/home/xinkai/data/ped2/ped2/testing/chunked_samples_frame/chunked_samples_frame_00.pkl"
    training_stat_path = "/home/xinkai/TCMP-ped2/ckpt_flow/training_stats.npy-2"

    with torch.no_grad():
        auc, aupr, eer = evaluate(config, args.flowmodel_save_path, args.framemodel_save_path,
                       testing_chunked_flow_samples_file, testing_chunked_frame_samples_file,
                       training_stat_path, suffix="best")

        print("auc:", auc)
        print("aupr:", aupr)
        print("eer:", eer)

