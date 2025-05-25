import os
import copy
import torch
import json
import logging
import warnings

import numpy as np
import yaml
from termcolor import colored

from tqdm import tqdm
from torch.optim import Adam

from utils.hignn_dataset import build_dataset
from utils.hignn_dataset import build_dataset_from_idx,build_loader
from HiGNN.source.train import parse_args

from parser_args import get_args
from tensorboardX import SummaryWriter
from chemprop.data import StandardScaler
from utils.dataset import Seq2seqDataset, get_data, split_data, MoleculeDataset, InMemoryDataset, load_npz_to_data_list
from utils.evaluate import eval_rocauc, eval_rmse
from torch.utils.data import BatchSampler, RandomSampler, DataLoader
from build_vocab import WordVocab
from chemprop.nn_utils import NoamLR
from chemprop.features import mol2graph, get_atom_fdim, get_bond_fdim
from chemprop.data.utils import get_class_sizes
from models_lib.multi_modal import Multi_modal
from featurizers.gem_featurizer import GeoPredTransformFn
import datetime
from yacs.config import CfgNode as CN

def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))


def load_smiles_to_dataset(data_path):
    """tbd"""
    data_list = []
    with open(data_path, 'r') as f:
        tmp_data_list = [line.strip() for line in f.readlines()]
        tmp_data_list = tmp_data_list[1:]
    data_list.extend(tmp_data_list)
    dataset = InMemoryDataset(data_list)
    return dataset

def cfg_to_dict(cfg_node):
    """递归将CfgNode对象转换为字典。"""
    config_dict = {}
    for k, v in cfg_node.items():
        if isinstance(v, CN):  # 如果值是CfgNode类型，递归转换
            config_dict[k] = cfg_to_dict(v)
        else:
            config_dict[k] = v
    return config_dict

def config2logger(args,cfg,logger):
    # 将配置转为字典
    config_dict = cfg_to_dict(cfg)
    # 使用 YAML 格式输出到日志文件，以便于可读性
    logger.info("Configuration:\n" + yaml.dump(config_dict, default_flow_style=False))
    # 将 args 转为字典
    args_dict = vars(args)
    # 使用 YAML 格式输出到日志文件，以便于可读性
    logger.info("Training configuration:\n" + yaml.dump(args_dict, default_flow_style=False))



def get_logger(args, cfg):
    output_dir = './LOG/{}/seed_{}/'.format(args.dataset, args.seed)
    logs_dir = os.path.join(output_dir,'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    tensorboard_file = output_dir + "tensorboard/"
    writer = SummaryWriter(log_dir=tensorboard_file)

    logs_file = logs_dir + "/{}_seed{}_无数据增强_{}.log".format(
        args.dataset, args.seed, datetime.date.today()
    )

    # 创建或获取 logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置 logger 的级别为 DEBUG

    # 定义日志格式
    fmt = '[%(asctime)s]:%(levelname)-5s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(fmt, datefmt=datefmt)
    console_handler.setFormatter(console_formatter)

    # 文件输出
    file_handler = logging.FileHandler(logs_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(fmt, datefmt=datefmt)
    file_handler.setFormatter(file_formatter)

    # 添加 handler 到 logger
    if not logger.handlers:  # 确保不重复添加 handler
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger, writer


def get_device(args, logger):
    if (torch.cuda.is_available() and args.cuda):
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.empty_cache()
        logger.info("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        logger.info("Device set to : cpu")

    return device



def get_seq_data(args, datas, device):
    smiles = datas.smiles()
    vocab = WordVocab.load_vocab('./data/{}_vocab.pkl'.format(args.dataset))
    args.seq_input_dim = args.vocab_num = len(vocab)
    seq = Seq2seqDataset(list(np.array(smiles)), vocab=vocab, seq_len=args.seq_len, device=device)
    seq_data = torch.stack([temp[1] for temp in seq])

    return seq_data



def get_3d_data(args, datas):
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate

    data_3d = InMemoryDataset(datas.smiles())
    transform_fn = GeoPredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])
    if not os.path.exists('./data/{}/'.format(args.dataset)):
        data_3d.transform(transform_fn, num_workers=1)
        data_3d.save_data('./data/{}/'.format(args.dataset))
    else:
        data_3d = data_3d._load_npz_data_path('./data/{}/'.format(args.dataset))
        data_3d = InMemoryDataset(data_3d)

    return data_3d

