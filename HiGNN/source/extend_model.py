import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv,global_add_pool
from transformers import BertModel, BertTokenizer

from model import HiGNN


class SMILESTransformer(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(SMILESTransformer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)

    def forward(self, smiles_list):
        # SMILES Token化，并获得Token ID
        inputs = self.tokenizer(smiles_list, padding=True, truncation=True, return_tensors='pt')
        # 将Token ID传递给BERT
        outputs = self.bert(**inputs)
        # 选取每个输入序列 [CLS] 标记的隐藏状态
        smiles_embedding = outputs.last_hidden_state[:,0,:]
        return smiles_embedding


class ExtendedHiGNN(HiGNN):
    def __init__(self, bert_model_name='bert-base-uncased', *args, **kwargs):
        super(ExtendedHiGNN, self).__init__(*args, **kwargs)
        self.smiles_transformer = SMILESTransformer(bert_model_name)

    def forward(self, data):
        # 获取原模型的输出
        mol_vec,fra_vec = super(ExtendedHiGNN, self).forward(data)

        smiles_list = data.smiles
        if smiles_list is not None:
            ''' *** Freeze Bert + HiGNN ***'''
            smiles_vec = data.smiles_vec

            ''' *** Fine-tuning Bert + HiGNN *** '''
            # self.smiles_transformer.cpu()
            # smiles_emb = self.smiles_transformer(smiles_list)
            # smiles_vec = torch.tensor(smiles_emb).cuda()
            # print('******* SMILES处理好了 ********')
            # 特征融合
            vectors_concat = list()
            vectors_concat.append(mol_vec)
            vectors_concat.append(fra_vec)
            vectors_concat.append(smiles_vec)
            out = torch.cat(vectors_concat, 1)

            # molecule-fragment contrastive
            if self.cl:
                out = F.dropout(out, p=self.dropout, training=self.training)
                return self.out(out), self.lin_project(mol_vec).relu_(), self.lin_project(fra_vec).relu_()
            else:
                out = F.dropout(out, p=self.dropout, training=self.training)
                return self.out(out)

        else:
            assert self.cl is False
            out = F.dropout(mol_vec, p=self.dropout, training=self.training)
            return self.out(out)



def build_model_ExtendedHiGNN(cfg):
    model = ExtendedHiGNN(in_channels=46,
                  hidden_channels=cfg.MODEL.HID,
                  out_channels=cfg.MODEL.OUT_DIM,
                  edge_dim=10,
                  num_layers=cfg.MODEL.DEPTH,
                  dropout=cfg.MODEL.DROPOUT,
                  slices=cfg.MODEL.SLICES,
                  f_att=cfg.MODEL.F_ATT,
                  r=cfg.MODEL.R,
                  brics=cfg.MODEL.BRICS,
                  cl=cfg.LOSS.CL_LOSS, )

    return model



































