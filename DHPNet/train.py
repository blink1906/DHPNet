import gc
import os
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import optim
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import shutil
from eval import evaluate
from tqdm import tqdm
from pre_processing.dataset import Chunked_frame_sample_dataset, Chunked_flow_sample_dataset, img_batch_tensor2numpy
from model.network import flow_net
from model.network_frame import frame_net

from model.loss import Gradient_Loss, Intensity_Loss
from utils.initialization_utils import weights_init_kaiming
from utils.vis_utils import visualize_sequences
from utils.model_utils import loader, saver, only_model_saver

torch.manual_seed(1936)

def train(config, training_chunked_flow_samples_dir, training_chunked_frame_samples_dir, testing_chunked_flow_samples_file, testing_chunked_frame_samples_file):
    paths_flow = dict(log_dir="%s/%s" % (config["log_root_flow"], config["exp_name"]),
                 ckpt_dir="%s/%s" % (config["ckpt_root_flow"], config["exp_name"]))

    paths_frame = dict(log_dir="%s/%s" % (config["log_root_frame"], config["exp_name"]),
                      ckpt_dir="%s/%s" % (config["ckpt_root_frame"], config["exp_name"]))

    os.makedirs(paths_flow["ckpt_dir"], exist_ok=True)
    os.makedirs(paths_frame["ckpt_dir"], exist_ok=True)

    batch_size = config["batchsize"]
    epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    device = config["device"]
    lr = config["lr"]
    training_chunk_frame_samples_files = sorted(os.listdir(training_chunked_frame_samples_dir))
    training_chunk_flow_samples_files = sorted(os.listdir(training_chunked_flow_samples_dir))

    grad_loss = Gradient_Loss(config["alpha"],
                              config["model_paras"]["frame_channels"] * config["model_paras"]["clip_pred"],
                              device).to(device)
    intensity_loss = Intensity_Loss(l_num=config["intensity_loss_norm"]).to(device)
    loss_func_mse = nn.MSELoss(reduction='none')

    model_flow = flow_net().to(device)

    model_frame = frame_net(proto_size=config["model_paras"]["proto_size"],
                 feature_dim=config["model_paras"]["feature_dim"],
                 proto_dim=config["model_paras"]["proto_dim"],
                 shink_thres=config["model_paras"]["shrink_thres"],
                 dropout=config["model_paras"]["drop_out"]).to(device)

    optimizer_flow = optim.Adam(model_flow.parameters(), lr=lr, eps=1e-7, weight_decay=0.0)
    scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, step_size=50, gamma=0.8)

    optimizer_frame = optim.Adam(model_frame.parameters(), lr=lr, eps=1e-7, weight_decay=0.0)
    scheduler_frame = torch.optim.lr_scheduler.StepLR(optimizer_frame, step_size=50, gamma=0.8)

    step = 0
    epoch_last = 0

    model_flow.apply(weights_init_kaiming)


    best_auc = -1
    best_aupr = -1
    best_eer = -1
    for epoch in range(epoch_last, epochs + epoch_last):

        for chunk_file_idx, chunk_file in enumerate(training_chunk_flow_samples_files):
            dataset = Chunked_flow_sample_dataset(os.path.join(training_chunked_flow_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            for idx, train_data in tqdm(enumerate(dataloader),
                                        desc="Training Epoch %d, Chunked File %d" % (epoch + 1, chunk_file_idx),
                                        total=len(dataloader)):
                model_flow.train()

                flow_tuple, _, pred_frame, _ = train_data
                flow_tuple = [flow.to(device) for flow in flow_tuple]

                out = model_flow(flow_tuple, mode="train")

                loss_flow = 0.0

                for output_flow, target_flow in zip(out["output_flow_tuple"], flow_tuple):
                    loss_flow += intensity_loss(output_flow, target_flow)

                loss_flow /= len(out["output_flow_tuple"])

                loss = config["lam_flow"] * loss_flow

                optimizer_flow.zero_grad()
                loss.backward()
                optimizer_flow.step()

                if step % config["logevery"] == config["logevery"] - 1:
                    print("[Step: {}/ Epoch: {}]: Loss: {:.4f} ".format(step + 1, epoch + 1, loss))
                    print("[Step: {}/ Epoch: {}]: loss_flow: {:.4f} ".format(step + 1, epoch + 1, loss_flow))

                step += 1
            del dataset

        scheduler_flow.step()

        for chunk_file_idx, chunk_file in enumerate(training_chunk_frame_samples_files):
            dataset = Chunked_frame_sample_dataset(os.path.join(training_chunked_frame_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=num_workers, shuffle=True)
            for idx, train_data in tqdm(enumerate(dataloader),
                                        desc="Training Epoch %d, Chunked File %d" % (epoch + 1, chunk_file_idx),
                                        total=len(dataloader)):
                model_frame.train()

                frame, pred_frame, _ = train_data
                frame = frame.to(device)

                out = model_frame(frame, mode="train")
                out["output_frame"] = out["output_frame"].to(device)
                out["frame_target"] = out["frame_target"].to(device)
                loss_frame_pred = torch.mean(loss_func_mse(out["output_frame"], out["frame_target"]))
                loss_frame_ortho = out["ortho_loss"]
                loss_frame_sep = out["separation_loss"]

                loss_frame = config["lam_frame_pred"] * loss_frame_pred + \
                             config["lam_frame_ortho"] * loss_frame_ortho + \
                             config["lam_frame_sep"] * loss_frame_sep

                loss = config["lam_frame"] * loss_frame

                optimizer_frame.zero_grad()
                loss.backward()
                optimizer_frame.step()

                if step % config["logevery"] == config["logevery"] - 1:
                    print("[Step: {}/ Epoch: {}]: Loss: {:.4f} ".format(step + 1, epoch + 1, loss))
                step += 1
            del dataset

        scheduler_frame.step()

        if epoch % config["saveevery"] == config["saveevery"] - 1:
            model_flow_save_path = os.path.join(paths_flow["ckpt_dir"], config["model_savename"])
            saver(model_flow.state_dict(), optimizer_flow.state_dict(), model_flow_save_path, epoch + 1, step, max_to_save=100)

            # computer training stats
            stats_flow_save_path = "/home/xinkai/TCMP-ped2/ckpt_flow/training_stats.npy-2"
            cal_flow_training_stats(config, "/home/xinkai/TCMP-ped2/ckpt_flow/model.pth-2", training_chunked_flow_samples_dir,
                               stats_flow_save_path)

            model_frame_save_path = os.path.join(paths_frame["ckpt_dir"], config["model_savename"])
            saver(model_frame.state_dict(), optimizer_frame.state_dict(), model_frame_save_path, epoch + 1, step,
                  max_to_save=100)

            # computer training stats
            # stats_frame_save_path = os.path.join(paths_frame["ckpt_dir"], "training_stats.npy-%d" % (epoch + 1))
            # cal_frame_training_stats(config, model_frame_save_path + "-%d" % (epoch + 1), training_chunked_frame_samples_dir,
            #                    stats_frame_save_path)

            with torch.no_grad():
                auc, aupr, eer = evaluate(config, "/home/xinkai/TCMP-ped2/ckpt_flow/model.pth-2", model_frame_save_path + "-%d" % (epoch + 1),
                               testing_chunked_flow_samples_file, testing_chunked_frame_samples_file,
                               stats_flow_save_path,suffix=str(epoch + 1))
                print(auc)
                if auc > best_auc:
                    best_auc = auc
                    only_model_saver(model_flow.state_dict(), os.path.join(paths_flow["ckpt_dir"], "best.pth"))
                    only_model_saver(model_frame.state_dict(), os.path.join(paths_frame["ckpt_dir"], "best.pth"))
                if aupr > best_aupr:
                    best_aupr = aupr
                if eer > best_eer:
                    best_eer = eer


    print("================ Best AUC %.4f ================" % best_auc)
    print("================ Best AUPR %.4f ================" % best_aupr)
    print("================ Best EER %.4f ================" % best_eer)


def cal_flow_training_stats(config, ckpt_path, training_chunked_samples_dir, stats_save_path):
    device = config["device"]
    model = flow_net().to(device).eval()

    # load weights
    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)
    # print("load pre-trained success!")

    score_func = nn.MSELoss(reduction="none")
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    # correspondence_stats = []
    flow_training_stats = []
    frame_training_stats = []

    print("=========Forward pass for training stats ==========")
    with torch.no_grad():

        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_flow_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=128, num_workers=0, shuffle=False)

            for idx, data in tqdm(enumerate(dataloader),
                                  desc="Training stats calculating, Chunked File %02d" % chunk_file_idx,
                                  total=len(dataloader)):
                flow_tuple, _, pred_frame, _ = data
                flow_tuple = [flow.to(device) for flow in flow_tuple]

                out = model(flow_tuple, mode="test")

                temp_loss_flow = None

                for output_flow, target_flow in zip(out["output_flow_tuple"], flow_tuple):
                    loss_flow = score_func(output_flow, target_flow).cpu().data.numpy()
                    if temp_loss_flow is None:
                        temp_loss_flow = loss_flow
                    else:
                        temp_loss_flow += loss_flow

                loss_flow = temp_loss_flow / len(out["output_flow_tuple"])

                flow_scores = np.sum(np.sum(np.sum(loss_flow, axis=3), axis=2), axis=1)
                flow_training_stats.append(flow_scores)
            del dataset
            gc.collect()

    print("=========Forward pass for training stats done!==========")
    flow_training_stats = np.concatenate(flow_training_stats, axis=0)

    training_stats = dict(flow_training_stats=flow_training_stats)
    # save to file
    torch.save(training_stats, stats_save_path)

def cal_frame_training_stats(config, ckpt_path, training_chunked_samples_dir, stats_save_path):
    device = config["device"]
    model = frame_net(proto_size=config["model_paras"]["proto_size"],
                 feature_dim=config["model_paras"]["feature_dim"],
                 proto_dim=config["model_paras"]["proto_dim"],
                 shink_thres=config["model_paras"]["shrink_thres"],
                 dropout=config["model_paras"]["drop_out"]).to(device).eval()

    # load weights
    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)
    # print("load pre-trained success!")

    score_func = nn.MSELoss(reduction="none")
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    frame_training_stats = []

    print("=========Forward pass for training stats ==========")
    with torch.no_grad():

        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_frame_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)

            for idx, data in tqdm(enumerate(dataloader),
                                  desc="Training stats calculating, Chunked File %02d" % chunk_file_idx,
                                  total=len(dataloader)):
                frame, pred_frame, _ = data

                out = model(frame, mode="test")
                out["output_frame"] = out["output_frame"].to(device)
                out["frame_target"] = out["frame_target"].to(device)
                loss_func_mse = nn.MSELoss(reduction='none')
                loss_frame_pred = torch.mean(loss_func_mse(out["output_frame"], out["frame_target"]))
                loss_frame_ortho = out["ortho_loss"]
                loss_frame_sep = out["separation_loss"]

                loss_frame = config["lam_frame_pred"] * loss_frame_pred + \
                             config["lam_frame_ortho"] * loss_frame_ortho + \
                             config["lam_frame_sep"] * loss_frame_sep


                frame_scores = np.sum(np.sum(np.sum(loss_frame, axis=3), axis=2), axis=1)
                frame_training_stats.append(frame_scores)
            del dataset
            gc.collect()

    print("=========Forward pass for training stats done!==========")
    frame_training_stats = np.concatenate(frame_training_stats, axis=0)

    training_stats = dict(frame_training_stats=frame_training_stats)
    # save to file
    torch.save(training_stats, stats_save_path)


if __name__ == '__main__':
    config = yaml.safe_load(open("./cfgs/cfg.yaml"))
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    training_chunked_flow_samples_dir = os.path.join(dataset_base_dir, dataset_name, "training/chunked_samples_flow")
    training_chunked_frame_samples_dir = os.path.join(dataset_base_dir, dataset_name, "training/chunked_samples_frame")
    testing_chunked_flow_samples_file = os.path.join(dataset_base_dir, dataset_name,
                                                "testing/chunked_samples_flow/chunked_samples_flow_00.pkl")
    testing_chunked_frame_samples_file = os.path.join(dataset_base_dir, dataset_name,
                                                     "testing/chunked_samples_frame/chunked_samples_frame_00.pkl")

    train(config, training_chunked_flow_samples_dir, training_chunked_frame_samples_dir, testing_chunked_flow_samples_file, testing_chunked_frame_samples_file)