import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import scipy.signal as signal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import ImageDraw, ImageFont

def add_scores_to_images(frame_scores, flow_scores, image_dir):
    for i, (frame_score, flow_score) in enumerate(zip(frame_scores, flow_scores)):
        img_path = os.path.join(image_dir, f'error_map_frame_{i + 1}.png')
        img = Image.open(img_path)

        plt.figure()
        plt.imshow(img)

        plt.text(160, 80, f'Frame score: {10 * frame_score:.2f}', color='white',
                 fontsize=14, bbox=dict(facecolor='white', alpha=0, pad=50))

        plt.axis('off')

        plt.savefig(img_path)

        plt.close()


def draw_roc_curve(fpr, tpr, auc, psnr_dir):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(psnr_dir, "auc.png"))
    plt.close()


def nonzero_intervals(vec):
    '''
    Find islands of non-zeros in the vector vec
    '''
    if len(vec) == 0:
        return []
    elif not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    tmp1 = (vec == 0) * 1
    tmp = np.diff(tmp1)
    edges, = np.nonzero(tmp)
    edge_vec = [edges + 1]

    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])
    edges = np.concatenate(edge_vec)
    return zip(edges[::2], edges[1::2])


def save_evaluation_curves(frame_scores, flow_scores, labels, curves_save_path, video_frame_nums):
    """
    Draw anomaly score curves for each video and the overall ROC figure.
    """
    if not os.path.exists(curves_save_path):
        os.mkdir(curves_save_path)

    frame_scores = frame_scores.flatten()
    flow_scores = flow_scores.flatten()

    labels = labels.flatten()

    frame_scores_each_video = {}
    flow_scores_each_video = {}
    scores_each_video = {}
    labels_each_video = {}

    start_idx = 0
    for video_id in range(len(video_frame_nums)):
        frame_scores_each_video[video_id] = frame_scores[start_idx:start_idx + video_frame_nums[video_id]]
        flow_scores_each_video[video_id] = flow_scores[start_idx:start_idx + video_frame_nums[video_id]]
        flow_scores_each_video[video_id] = signal.medfilt(flow_scores_each_video[video_id], kernel_size=17)
        scores_each_video[video_id] = 1 * flow_scores_each_video[video_id] + 18 * frame_scores_each_video[video_id]
        labels_each_video[video_id] = labels[start_idx:start_idx + video_frame_nums[video_id]]

        start_idx += video_frame_nums[video_id]

    truth = []
    preds = []

    for i in range(len(scores_each_video)):
        truth.append(labels_each_video[i])
        preds.append(scores_each_video[i])
    truth = np.concatenate(truth, axis=0)
    preds = np.concatenate(preds, axis=0)
    fpr, tpr, roc_thresholds = roc_curve(truth, preds, pos_label=1)
    fnr = 1 - tpr
    eer_thresholds = roc_thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    precision, recall, thresholds = precision_recall_curve(truth, preds)
    aupr = auc(recall, precision)
    auroc = auc(fpr, tpr)


    draw_roc_curve(fpr, tpr, auroc, curves_save_path)
    font = {'family': 'sans-serif',
            'color': 'black',
            'weight': 'normal',
            'size': 18,
            }

    for i in sorted(scores_each_video.keys()):
        plt.figure(figsize=(12.8, 2.4))

        x = range(0, len(scores_each_video[i]))
        plt.xlim([x[0], x[-1] + 5])

        plt.plot(x, scores_each_video[i], color="black", lw=2, label="Frame Anomaly Score")

        lb_one_intervals = nonzero_intervals(labels_each_video[i])
        for idx, (start, end) in enumerate(lb_one_intervals):
            plt.axvspan(start, end, alpha=0.5, color='green',
                        label="_" * idx + "Anomaly Intervals")

        plt.text(0.96, 0.2, f"AUROC: {100 * auroc:.2f}%", ha='right', va='top',
                 transform=plt.gca().transAxes, fontdict=font)

        # Add vertical text
        plt.text(0.03, 0.9, 'MPN:', color='black', fontsize=25, ha='left', va='top', transform=plt.gca().transAxes)

        plt.xticks([])
        plt.yticks([])

        plt.savefig(os.path.join(curves_save_path, "anomaly_curve_%d.png" % (i + 1)), dpi=100)
        plt.close()

    return auroc, aupr, eer


def save_flow_evaluation_curves(flow_scores, labels, curves_save_path, video_frame_nums):
    """
    Draw anomaly score curves for each video and the overall ROC figure.
    """
    if not os.path.exists(curves_save_path):
        os.mkdir(curves_save_path)

    flow_scores = flow_scores.flatten()
    labels = labels.flatten()

    flow_scores_each_video = {}
    scores_each_video = {}
    labels_each_video = {}

    start_idx = 0
    for video_id in range(len(video_frame_nums)):
        flow_scores_each_video[video_id] = flow_scores[start_idx:start_idx + video_frame_nums[video_id]]
        flow_scores_each_video[video_id] = signal.medfilt(flow_scores_each_video[video_id], kernel_size=17)
        scores_each_video[video_id] = 1 * flow_scores_each_video[video_id]
        labels_each_video[video_id] = labels[start_idx:start_idx + video_frame_nums[video_id]]

        start_idx += video_frame_nums[video_id]

    truth = []
    preds = []
    for i in range(len(scores_each_video)):
        truth.append(labels_each_video[i])
        preds.append(scores_each_video[i])

    truth = np.concatenate(truth, axis=0)
    preds = np.concatenate(preds, axis=0)
    fpr, tpr, roc_thresholds = roc_curve(truth, preds, pos_label=1)
    fnr = 1 - tpr
    eer_thresholds = roc_thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    precision, recall, thresholds = precision_recall_curve(truth, preds)
    aupr = auc(recall, precision)
    auroc = auc(fpr, tpr)

    # draw ROC figure
    draw_roc_curve(fpr, tpr, auroc, curves_save_path)
    for i in sorted(scores_each_video.keys()):
        plt.figure()

        x = range(0, len(scores_each_video[i]))
        plt.xlim([x[0], x[-1] + 5])

        plt.plot(x, scores_each_video[i], color="blue", lw=2, label="Anomaly Score")

        lb_one_intervals = nonzero_intervals(labels_each_video[i])
        for idx, (start, end) in enumerate(lb_one_intervals):
            plt.axvspan(start, end, alpha=0.5, color='red',
                        label="_" * idx + "Anomaly Intervals")

        plt.xlabel('Frames Sequence')
        plt.title('Test video #%d' % (i + 1))
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(curves_save_path, "anomaly_curve_%d.png" % (i + 1)))
        plt.close()

    return auroc, aupr, eer
