import torch.nn as nn
from transformers import BertModel, BertTokenizer


class SMILESTransformer(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(SMILESTransformer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)

        # 固定冻结Bert模型参数
        for param in self.bert.parameters():
            param.requires_grad = False


    def forward(self, smiles_list):
        # SMILES Token化，并获得Token ID
        inputs = self.tokenizer(smiles_list, padding=True, truncation=True, return_tensors='pt')
        # 将Token ID传递给BERT
        outputs = self.bert(**inputs)
        # 选取每个输入序列 [CLS] 标记的隐藏状态
        smiles_embedding = outputs.last_hidden_state[:,0,:]
        return smiles_embedding

