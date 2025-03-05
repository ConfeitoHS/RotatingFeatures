from typing import Dict

import numpy as np
import skimage
import torch
from einops import rearrange
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kmeans_pytorch import kmeans

from codebase.utils import utils


@ignore_warnings(category=ConvergenceWarning)
def apply_kmeans(
    opt: DictConfig, norm_rotating_output: torch.Tensor, gt_labels: torch.Tensor
) -> np.ndarray:
    """
    Apply k-means clustering to the reconstructed, normalized rotating features.

    Args:
        opt (DictConfig): Configuration options.
        norm_rotating_output (torch.Tensor): Normalized rotating features, shape (b, n, c, h, w).
        gt_labels (torch.Tensor): Ground-truth labels, shape (b, h, w).

    Returns:
        np.ndarray: Predicted labels.
    """
    num_clusters = opt.input.num_objects_per_img + 1
    norm_rotating_output = rearrange(
        norm_rotating_output.detach().cpu().numpy(), "b n c h w -> b h w (c n)"
    )
    pred_labels = np.zeros(
        (opt.input.batch_size, opt.input.image_size[0], opt.input.image_size[1])
    )

    # Run k-means on each image separately.
    for img_idx in range(opt.input.batch_size):
        norm_rotating_output_img = norm_rotating_output[img_idx]

        if opt.evaluation.mask_overlap == 1:
            # Remove overlap areas before applying k-means.
            label_idx = np.where(gt_labels[img_idx] != -1)
            norm_rotating_output_img = norm_rotating_output_img[label_idx]
        else:
            norm_rotating_output_img = rearrange(
                norm_rotating_output_img, "h w c -> (h w) c"
            )

        # Run k-means.
        k_means = KMeans(n_clusters=num_clusters, random_state=opt.seed, n_init=10).fit(
            norm_rotating_output_img
        )

        if opt.evaluation.mask_overlap == 1:
            # Create result image: fill in with predicted labels & assign "none" label to overlap areas.
            cluster_img = np.zeros(opt.input.image_size) + num_clusters
            cluster_img[label_idx] = k_means.labels_
        else:
            cluster_img = rearrange(
                k_means.labels_,
                "(h w) -> h w",
                h=opt.input.image_size[0],
                w=opt.input.image_size[1],
            )

        pred_labels[img_idx] = cluster_img

    return pred_labels


def apply_tsne(
    opt: DictConfig, norm_rotating_output: torch.Tensor, gt_labels: torch.Tensor
):

    pred_labels = apply_kmeans(opt, norm_rotating_output, gt_labels)

    mbo_args = np.array(
        [
            mean_best_overlap_single_sample(
                gt_labels["pixelwise_instance_labels"][b_idx].detach().cpu().numpy(),
                pred_labels[b_idx],
            )[1]
            for b_idx in range(opt.input.batch_size)
        ]
    )

    mbo_args = np.concatenate(
        ((6 - mbo_args.sum(1))[:, None], mbo_args, np.zeros_like(mbo_args)), axis=1
    )
    from tqdm import tqdm

    pred_mino = np.array(
        [
            (
                np.concatenate(
                    (
                        gt_labels["shape"][b_idx].detach().cpu().numpy(),
                        np.zeros_like(gt_labels["shape"][b_idx].detach().cpu().numpy()),
                    )
                )
            )[mbo_args[b_idx][mbo_args[b_idx]][pred_labels[b_idx].astype(int)]]
            for b_idx in tqdm(range(opt.input.batch_size))
        ]
    )
    # print(pred_labels.astype(int))
    # print(pred_mino)
    # print(gt_labels["shape"][0])

    # ex) mbo_args[0] is best match pred label for background
    # ex) mbo_args[1] is best match pred label for label 1 in GT

    # model = TSNE(verbose=1)
    # model = PCA(n_components=2)
    model = LinearDiscriminantAnalysis(n_components=2)
    norm_rotating_output = rearrange(
        norm_rotating_output.detach().cpu().numpy(), "b n c h w -> (b h w) (c n)"
    )

    pred_mino = rearrange(pred_mino, "b h w -> (b h w)")
    idx = np.logical_not(pred_mino == 0)
    return (
        model.fit_transform(norm_rotating_output[idx], pred_mino[idx]),
        pred_mino[idx],
    )


def apply_kmeans_gpu(
    opt: DictConfig, norm_rotating_output: torch.Tensor, gt_labels: torch.Tensor
) -> np.ndarray:
    """
    Apply k-means clustering to the reconstructed, normalized rotating features.

    Args:
        opt (DictConfig): Configuration options.
        norm_rotating_output (torch.Tensor): Normalized rotating features, shape (b, n, c, h, w).
        gt_labels (torch.Tensor): Ground-truth labels, shape (b, h, w).

    Returns:
        np.ndarray: Predicted labels.
    """
    num_clusters = opt.input.num_objects_per_img + 1

    norm_rotating_output = rearrange(
        norm_rotating_output.detach(), "b n c h w -> b h w (c n)"
    )
    pred_labels = torch.zeros(
        (opt.input.batch_size, opt.input.image_size[0], opt.input.image_size[1])
    )

    if opt.evaluation.mask_overlap == 1:
        # Remove overlap areas before applying k-means.
        label_idx = torch.where(gt_labels != -1)

        norm_rotating_output = norm_rotating_output[label_idx]
    else:
        norm_rotating_output = rearrange(norm_rotating_output, "b h w c -> b (h w) c")
    for img_idx in range(opt.input.batch_size):
        labels_, __annotations__ = kmeans(
            X=norm_rotating_output[img_idx],
            num_clusters=num_clusters,
            device=torch.device("cuda:0"),
        )

        if opt.evaluation.mask_overlap == 1:
            # Create result image: fill in with predicted labels & assign "none" label to overlap areas.
            cluster_img = torch.zeros_like(pred_labels) + num_clusters
            cluster_img[label_idx] = labels_.float()
        else:
            cluster_img = rearrange(
                labels_,
                "(h w) -> h w",
                h=opt.input.image_size[0],
                w=opt.input.image_size[1],
            )

        pred_labels[img_idx] = cluster_img

    return pred_labels.cpu().numpy()


def calc_ari_score(
    opt: DictConfig, gt_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """
    Calculate Adjusted Rand Index (ARI) score for object discovery evaluation.

    Args:
        opt (DictConfig): Configuration options.
        gt_labels (np.ndarray): Ground truth labels, shape ((b, h, w)).
        pred_labels (np.ndarray): Predicted labels, shape (b, h, w).

    Returns:
        float: ARI score.
    """
    ari = 0
    for idx in range(opt.input.batch_size):
        # Remove "ignore" (-1) and background (0) gt_labels.
        area_to_eval = np.where(gt_labels[idx] > 0)

        ari += adjusted_rand_score(
            gt_labels[idx][area_to_eval], pred_labels[idx][area_to_eval]
        )
    return ari / opt.input.batch_size


def compute_iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) matrix between ground truth and predicted labels.

    Args:
        gt_labels (np.ndarray): Ground truth labels, shape (m, h, w).
        pred_labels (np.ndarray): Predicted labels, shape (o, h, w).

    Returns:
        np.ndarray: IoU matrix, shape (m, o).
    """
    intersection = np.logical_and(
        gt_labels[:, None, :, :], pred_labels[None, :, :, :]
    ).sum(axis=(2, 3))
    union = np.logical_or(gt_labels[:, None, :, :], pred_labels[None, :, :, :]).sum(
        axis=(2, 3)
    )
    return intersection / (union + 1e-9)


def mean_best_overlap_single_sample(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """
    Compute the Mean Best Overlap (MBO) for a single sample between ground truth and predicted labels.

    Args:
        gt_labels (np.ndarray): Ground truth labels, shape (h, w).
        pred_labels (np.ndarray): Predicted labels, shape (h, w).

    Returns:
        float: MBO score for the sample.
    """
    unique_gt_labels = np.unique(gt_labels)
    # Remove "ignore" (-1) label.
    unique_gt_labels = unique_gt_labels[unique_gt_labels != -1]

    # Mask areas with "ignore" gt_labels in pred_labels.
    pred_labels[np.where(gt_labels < 0)] = -1

    # Ignore background (0) gt_labels.
    unique_gt_labels = unique_gt_labels[unique_gt_labels != 0]

    if len(unique_gt_labels) == 0:
        return -1  # If no gt_labels left, skip this element.

    unique_pred_labels = np.unique(pred_labels)

    # Remove "ignore" (-1) label.
    unique_pred_labels = unique_pred_labels[unique_pred_labels != -1]

    gt_masks = np.equal(gt_labels[None, :, :], unique_gt_labels[:, None, None])
    pred_masks = np.equal(pred_labels[None, :, :], unique_pred_labels[:, None, None])

    iou_matrix = compute_iou_matrix(gt_masks, pred_masks)
    best_iou = np.max(iou_matrix, axis=1)
    best_iou_arg = np.argmax(iou_matrix, axis=1)
    return np.mean(best_iou), best_iou_arg


def calc_mean_best_overlap(
    opt: DictConfig, gt_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """
    Calculate the Mean Best Overlap (MBO) for a batch of ground truth and predicted labels.

    Args:
        opt (DictConfig): Configuration options.
        gt_labels (np.ndarray): Ground truth labels, shape (b, h, w).
        pred_labels (np.ndarray): Predicted labels, shape (b, h, w).

    Returns:
        float: MBO score for the batch.
    """
    mean_best_overlap = np.array(
        [
            mean_best_overlap_single_sample(gt_labels[b_idx], pred_labels[b_idx])[0]
            for b_idx in range(opt.input.batch_size)
        ]
    )

    if np.any(mean_best_overlap != -1):
        return np.mean(mean_best_overlap[mean_best_overlap != -1])
    else:
        return 0.0


def run_object_discovery_evaluation(
    opt: DictConfig, pred_labels: np.ndarray, gt_labels: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Run object discovery evaluation and calculate metrics.

    Args:
        opt (DictConfig): Configuration options.
        pred_labels (np.ndarray): Predicted labels, shape (b, h, w).
        gt_labels (Dict[str, torch.Tensor]): Ground truth labels, each of shape (b, h, w).

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    pred_labels = resize_pred_labels(
        opt, pred_labels, gt_labels["pixelwise_instance_labels"]
    )
    gt_labels = utils.tensor_dict_to_numpy(gt_labels, dtype=np.int32)

    metrics = {
        "ARI": calc_ari_score(
            opt, gt_labels["pixelwise_instance_labels"], pred_labels.copy()
        ),
        "MBO_i": calc_mean_best_overlap(
            opt, gt_labels["pixelwise_instance_labels"], pred_labels.copy()
        ),
    }

    if "pixelwise_class_labels" in gt_labels:
        metrics["MBO_c"] = calc_mean_best_overlap(
            opt,
            gt_labels["pixelwise_class_labels"],
            pred_labels,
        )
    return metrics


def resize_pred_labels(
    opt: DictConfig, pred_labels: np.ndarray, gt_labels: np.ndarray
) -> np.ndarray:
    """
    Resize predicted labels to match the shape of ground truth labels and optionally smooth them.

    Args:
        opt (DictConfig): Configuration options.
        pred_labels (np.ndarray): Predicted labels, shape (b, h, w).
        gt_labels (np.ndarray): Ground truth labels, shape (b, h, w).

    Returns:
        np.ndarray: Resized and optionally smoothed predicted labels.
    """
    if pred_labels.shape == gt_labels.shape:
        return pred_labels

    # Resize pred_labels to shape of gt_labels.
    resized_pred_labels = (
        torch.nn.functional.interpolate(
            torch.Tensor(pred_labels)[:, None],
            size=(
                gt_labels.shape[1],
                gt_labels.shape[2],
            ),
            mode="nearest",
        )[:, 0]
        .numpy()
        .astype(np.uint8)
    )

    if opt.evaluation.smooth_labels:
        # Smooth out predicted labels by applying mode filter.
        disk_size = gt_labels.shape[1] // pred_labels.shape[1]
        for i in range(resized_pred_labels.shape[0]):
            resized_pred_labels[i] = skimage.filters.rank.modal(
                skimage.util.img_as_ubyte(resized_pred_labels[i]),
                skimage.morphology.disk(disk_size),
            )

    return resized_pred_labels
