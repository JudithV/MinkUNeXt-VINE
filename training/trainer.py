# Code adapted from MinkLoc3Dv2 repo: https://github.com/jac99/MinkLoc3Dv2.git
# Institute for Engineering Research of Elche (I3E)
# Automation, Robotics and Computer Vision lab (ARCV)
# Author: Judith Vilella Cantos

import os
import numpy as np
import torch
from config import PARAMS
import tqdm
import pathlib
import wandb
from losses.contrastive_loss import BatchHardContrastiveLossWithMasks
from losses.matryoshka import MatryoshkaLoss
from datasets.dataset_utils import make_dataloaders
from pnv_evaluate import evaluate, print_eval_stats, pnv_write_eval_stats
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity


def print_global_stats(phase, stats):
    s = f"{phase}  loss: {stats['loss']:.4f}   embedding norm: {stats['avg_embedding_norm']:.3f}  "
    if 'num_triplets' in stats:
        s += f"Triplets (all/active): {stats['num_triplets']:.1f}/{stats['num_non_zero_triplets']:.1f}  " \
             f"Mean dist (pos/neg): {stats['mean_pos_pair_dist']:.3f}/{stats['mean_neg_pair_dist']:.3f}   "
    if 'positives_per_query' in stats:
        s += f"#positives per query: {stats['positives_per_query']:.1f}   "
    if 'best_positive_ranking' in stats:
        s += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
    if 'recall' in stats:
        s += f"Recall@1: {stats['recall'][1]:.4f}   "
    if 'ap' in stats:
        s += f"AP: {stats['ap']:.4f}   "

    print(s)


def print_stats(phase, stats):
    print_global_stats(phase, stats['global'])


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def training_step(global_iter, model, phase, device, optimizer, loss_fn):
    assert phase in ['train', 'val']

    batch, positives_mask, negatives_mask = next(global_iter)
    batch = {e: batch[e].to(device) for e in batch}
    if phase == 'train':
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()
    distance = LpDistance(normalize_embeddings=PARAMS.normalize_embeddings)
    loss_fn_cluster = BatchHardContrastiveLossWithMasks(pos_margin=PARAMS.pos_margin, neg_margin=PARAMS.neg_margin, distance=distance)
    loss_fn_matryoshka = MatryoshkaLoss(loss_fn, dims=[64, 128, 192], weights=[1.0, 0.5, 0.25])
    with torch.set_grad_enabled(phase == 'train'):
        y = model(batch)

        stats = model.stats.copy() if hasattr(model, 'stats') else {}

        embeddings = y['global']

        #loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)
        loss, temp_stats = loss_fn_matryoshka(embeddings, positives_mask, negatives_mask)
        #loss, temp_stats = loss_fn_cluster(embeddings, positives_mask, negatives_mask)
        if PARAMS.clustering_head:
            distance = LpDistance(normalize_embeddings=PARAMS.normalize_embeddings)
            loss_fn_cluster = BatchHardContrastiveLossWithMasks(pos_margin=PARAMS.pos_margin, neg_margin=PARAMS.neg_margin, distance=distance)
            clustering = y['clustering_emb'] # y['clustering_emb']
            if clustering.shape[0] > PARAMS.cluster_batch_size:
                idx = torch.randperm(clustering.shape[0])[:PARAMS.cluster_batch_size]
                clustering_sub = clustering[idx]
                positives_mask_sub = positives_mask[idx][:, idx]
                negatives_mask_sub = negatives_mask[idx][:, idx]
            else:
                clustering_sub = clustering
                positives_mask_sub = positives_mask
                negatives_mask_sub = negatives_mask


            loss_clust, _ = loss_fn(clustering, positives_mask, negatives_mask) # loss_fn_cluster
            lambda_clust = PARAMS.clustering_importance
            loss = loss + lambda_clust * loss_clust
        temp_stats = tensors_to_numbers(temp_stats)
        stats.update(temp_stats)
        if phase == 'train':
            loss.backward()
            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return stats


def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn):
    # Training step using multistaged backpropagation algorithm as per:
    # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
    # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
    # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
    # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774

    assert phase in ['train', 'val']
    batch, positives_mask, negatives_mask = next(global_iter)
    distance = LpDistance(normalize_embeddings=PARAMS.normalize_embeddings)
    loss_fn_cluster = BatchHardContrastiveLossWithMasks(pos_margin=PARAMS.pos_margin, neg_margin=PARAMS.neg_margin, distance=distance)
    loss_fn_matryoshka = MatryoshkaLoss(loss_fn, dims=[64, 128, 192], weights=[1.0, 0.5, 0.25])
    if phase == 'train':
        model.train()
    else:
        model.eval()

    # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
    # In training phase network is in the train mode to update BatchNorm stats
    embeddings_l = []
    if PARAMS.clustering_head:
        clustering_l = []
    
    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device) for e in minibatch}
            y = model(minibatch)
            embeddings_l.append(y['global'])
            if PARAMS.clustering_head:
                clustering_l.append(y['clustering_emb']) # y['clustering_emb']

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    embeddings = torch.cat(embeddings_l, dim=0)
    
    embeddings_grad = None
    clustering_grad = None
    clustering_logits_grad = None

    if PARAMS.clustering_head:
        clustering = torch.cat(clustering_l, dim=0)

    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
            if PARAMS.clustering_head:
                clustering.requires_grad_(True)

        loss, stats = loss_fn_matryoshka(embeddings, positives_mask, negatives_mask)
        stats = tensors_to_numbers(stats)

        total_loss = loss
        
        if PARAMS.clustering_head:
            loss_clust, _ = loss_fn(clustering, positives_mask, negatives_mask)
            total_loss += (PARAMS.clustering_importance * loss_clust)

        if phase == 'train':
            total_loss.backward()
            
            embeddings_grad = embeddings.grad
            if PARAMS.clustering_head:
                clustering_grad = clustering.grad

    embeddings_l, embeddings, loss = None, None, None
    if PARAMS.clustering_head:
        clustering_l, clustering = None, None

    # ---------------- Stage 3: Backpropagation ----------------
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                y = model(minibatch)
                
                # Output Global
                embeddings_mb = y['global']
                minibatch_size = embeddings_mb.shape[0]
                
                if embeddings_grad is not None:
                    retain_graph = PARAMS.clustering_head
                    embeddings_mb.backward(gradient=embeddings_grad[i: i+minibatch_size], retain_graph=retain_graph)

                if PARAMS.clustering_head:
                    clustering_mb = y['clustering_emb'] # y['clustering_emb']
                    
                    if clustering_grad is not None:
                        clustering_mb.backward(gradient=clustering_grad[i: i+minibatch_size], retain_graph=PARAMS.clustering_head)

                i += minibatch_size

            optimizer.step()

    torch.cuda.empty_cache()
    return stats, 0, 0


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
