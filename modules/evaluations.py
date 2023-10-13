"""
This script was modified from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
"""
import os
import cv2
import bcolz
import numpy as np
import tqdm
from sklearn.model_selection import KFold
from scipy.spatial import distance
import matplotlib.pyplot as plt
from .utils import l2_norm
from scipy.signal import correlate

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_val_data(data_path):
    """get validation data"""
    lfw, lfw_issame = get_val_pair(data_path, 'lfw_align_112/lfw')
    agedb_30, agedb_30_issame = get_val_pair(data_path,
                                             'agedb_align_112/agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_align_112/cfp_fp')

    return lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame


def ccrop_batch(imgs):
    assert len(imgs.shape) == 4
    resized_imgs = np.array([cv2.resize(img, (128, 128)) for img in imgs])
    ccropped_imgs = resized_imgs[:, 8:-8, 8:-8, :]

    return ccropped_imgs


def hflip_batch(imgs):
    assert len(imgs.shape) == 4
    return imgs[:, :, ::-1, :]


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    
    return tpr, fpr, acc, tp, fp, tn, fn

def ssim(x,y):
  ux = x.mean()
  uy = y.mean()
  sx = (((x - ux)**2).mean())**0.5
  sy = (((y - uy)**2).mean())**0.5
  sxy = ((x - ux)*(y - uy)).mean()
  ssim_ = ((2*ux*uy+0.0001)*(2*sxy+0.0002))/((ux**2+uy**2+0.0001)*(sx**2+sy**2+0.0002))
  return ssim_
def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame,
                  nrof_folds=10, sim="diff"):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    fps = np.zeros((nrof_folds, nrof_thresholds))
    tps = np.zeros((nrof_folds, nrof_thresholds))
    fns = np.zeros((nrof_folds, nrof_thresholds))
    tns = np.zeros((nrof_folds, nrof_thresholds))

    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    if sim == "diff":
      diff = np.subtract(embeddings1, embeddings2)
      dist = np.sum(np.square(diff), 1)
    elif sim == "corr":
      diff = np.array([correlate(embeddings1[i,:], embeddings2[i,:], method='auto') for i in range(embeddings1.shape[0])])
      dist = 1-np.max(diff,axis=-1)
    elif sim == "fft_corr":
      diff = np.array([correlate(embeddings1[i,:], embeddings2[i,:], method='fft') for i in range(embeddings1.shape[0])])
      dist = 1-np.max(diff,axis=-1)
    elif sim == "direct_corr":
      diff = np.array([correlate(embeddings1[i,:], embeddings2[i,:], method='direct') for i in range(embeddings1.shape[0])])
      dist = 1-np.max(diff,axis=-1)
    elif sim == "cos":
      diff = np.array([distance.cosine(embeddings1[i,:], embeddings2[i,:]) for i in range(embeddings1.shape[0])])
      dist = diff
    elif sim=="ssim":
      diff = np.array([ssim(embeddings1[i,:], embeddings2[i,:]) for i in range(embeddings1.shape[0])])
      dist = 1-diff


    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx],_,_,_,_ = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, tps[fold_idx, threshold_idx], fps[fold_idx, threshold_idx], tns[fold_idx, threshold_idx], fns[fold_idx, threshold_idx] = \
                calculate_accuracy(threshold,
                                   dist[test_set],
                                   actual_issame[test_set])
        _, _, accuracy[fold_idx], _, _, _, _ = calculate_accuracy(
            thresholds[best_threshold_index],
            dist[test_set],
            actual_issame[test_set])
        

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    print(f"tp: {np.mean(tps, 0).sum()}, fp: {np.mean(fps, 0).sum()}, tn: {np.mean(tns, 0).sum()}, fn: {np.mean(fns, 0).sum()}")
    p = np.mean(tps, 0).sum()/(np.mean(tps, 0).sum()+np.mean(fps, 0).sum())
    r = np.mean(tps, 0).sum()/(np.mean(tps, 0).sum()+np.mean(fns, 0).sum())
    f1 = 2*p*r/(p+r)
    eff = np.mean(tns, 0).sum()/(np.mean(tns, 0).sum()+np.mean(fps, 0).sum())
    print(f"f1: {f1}, pre: {p}, recall: {r}, , eff: {eff}")
    data = [[np.mean(tps, 0).sum(), np.mean(fps, 0).sum()],[np.mean(fns, 0).sum(), np.mean(tns, 0).sum()]]
    heatmap = plt.pcolor(data)
    plt.colorbar(heatmap)
    plt.show()
    return tpr, fpr, accuracy, best_thresholds


def evaluate(embeddings, actual_issame, sim, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
        nrof_folds=nrof_folds, sim=sim)

    return tpr, fpr, accuracy, best_thresholds


def perform_val(embedding_size, batch_size, model,
                carray, issame, sim, nrof_folds=10, is_ccrop=False, is_flip=True):
    """perform val"""
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm.tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1]) * 0.5 + 0.5
        batch = batch[:, :, :, ::-1]  # convert BGR to RGB

        if is_ccrop:
            batch = ccrop_batch(batch)
        if is_flip:
            fliped = hflip_batch(batch)
            emb_batch = model(batch) + model(fliped)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            emb_batch = model(batch)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(
        embeddings, issame, sim, nrof_folds)

    return accuracy.mean(), best_thresholds.mean(), tpr, fpr
