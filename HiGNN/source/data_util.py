import os
from random import Random

import numpy as np
import torch
from torch_geometric.data import DataLoader
from dataset import MolDataset
from utils import get_task_names


def load_dataset_random(path, dataset, seed, task_type, tasks=None, logger=None):
    save_path = path + 'processed/merge_train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test
    pyg_dataset = MolDataset(root=path, dataset=dataset, task_type=task_type, tasks=tasks, logger=logger)

    # 生成随机数
    random = Random(seed)
    indices = list(range(len(pyg_dataset)))
    train_indices1 = indices[14134:]  # 非天然
    random_indices = indices[:14133]  # 天然
    random.shuffle(random_indices)
    train_size = int(0.7 * len(random_indices))
    val_size = int(0.15 * len(random_indices))
    test_size = len(random_indices) - train_size - val_size

    trn_id = random_indices[:train_size]
    trn_id = trn_id + train_indices1
    val_id = random_indices[train_size:(train_size + val_size)]
    test_id = random_indices[(train_size + val_size):]

    trn = pyg_dataset[torch.LongTensor(trn_id)]
    val = pyg_dataset[torch.LongTensor(val_id)]
    test = pyg_dataset[torch.LongTensor(test_id)]

    logger.info(f'Total smiles = {len(pyg_dataset):,} | '
                f'train smiles = {train_size:,} | '
                f'val smiles = {val_size:,} | '
                f'test smiles = {test_size:,}')

    # 处理数据集中不同类别的样本数量不平衡的问题
    assert task_type == 'classification' or 'regression'
    if task_type == 'classification':
        weights = []
        for i in range(len(tasks)):
            validId = np.where((pyg_dataset.data.y[:, i] == 0) | (pyg_dataset.data.y[:, i] == 1))[0]
            pos_len = (pyg_dataset.data.y[:, i][validId].sum()).item()
            neg_len = len(pyg_dataset.data.y[:, i][validId]) - pos_len
            # 根据正例和负例的数量计算权重:如果负例比正例多，则正例的权重更高，反之亦然
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
        trn.weights = weights

    else:
        trn.weights = None

    torch.save([trn, val, test], save_path)
    return trn, val, test






def build_dataset(cfg, logger):

    cfg.defrost()
    task_name = get_task_names(os.path.join(cfg.DATA.DATA_PATH, 'raw/{}.csv'.format(cfg.DATA.DATASET)))
    if cfg.DATA.TASK_TYPE == 'classification':
        out_dim = 2 * len(task_name)
    elif cfg.DATA.TASK_TYPE == 'regression':
        out_dim = len(task_name)
    else:
        raise Exception('Unknown task type')
    opts = ['DATA.TASK_NAME', task_name, 'MODEL.OUT_DIM', out_dim]
    cfg.defrost()
    cfg.merge_from_list(opts)
    cfg.freeze()

    if cfg.DATA.SPLIT_TYPE == 'random':
        train_dataset, valid_dataset, test_dataset = load_dataset_random(cfg.DATA.DATA_PATH,
                                                                         cfg.DATA.DATASET,
                                                                         cfg.SEED,
                                                                         cfg.DATA.TASK_TYPE,
                                                                         cfg.DATA.TASK_NAME,
                                                                         logger)

    elif cfg.DATA.SPLIT_TYPE == 'scaffold':
        return
        # train_dataset, valid_dataset, test_dataset = load_dataset_scaffold(cfg.DATA.DATA_PATH,
        #                                                                    cfg.DATA.DATASET,
        #                                                                    cfg.SEED,
        #                                                                    cfg.DATA.TASK_TYPE,
        #                                                                    cfg.DATA.TASK_NAME,
        #                                                                    logger)

    elif cfg.DATA.SPLIT_TYPE == 'noise':
        return
        # train_dataset, valid_dataset, test_dataset = load_dataset_noise(cfg.DATA.DATA_PATH,
        #                                                                 cfg.DATA.DATASET,
        #                                                                 cfg.SEED,
        #                                                                 cfg.DATA.TASK_TYPE,
        #                                                                 cfg.DATA.TASK_NAME,
        #                                                                 cfg.DATA.RATE,
        #                                                                 logger)

    else:
        raise Exception('Unknown dataset split type')

    return train_dataset, valid_dataset, test_dataset


def build_loader_mergedataset(cfg, logger):
    train_dataset, valid_dataset, test_dataset = build_dataset(cfg, logger)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.DATA.BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.DATA.BATCH_SIZE)

    weights = train_dataset.weights

    return train_dataloader, valid_dataloader, test_dataloader, weights


























