import torch
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertModel, 
    AutoTokenizer,
    AutoModel,
    AutoModelForMultipleChoice,
)
from sentence_transformers import SentenceTransformer

def get_bert_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=512)
    model = AutoModel.from_pretrained(path)
    return tokenizer, model

def get_distilbert_model(path: str='distilbert-base-uncased'):
    tokenizer = DistilBertTokenizerFast.from_pretrained(path)
    model = DistilBertModel.from_pretrained(path)
    return tokenizer, model

def get_bert_mc_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=512)
    model = AutoModelForMultipleChoice.from_pretrained(path)
    return tokenizer, model

def get_sentence_bert_model(path: str='sentence-transformers/all-MiniLM-L12-v2'):
    model = SentenceTransformer(path)
    return model

class BertClass(torch.nn.Module):
    def __init__(self, 
            model,
            output_dim=2
        ):
        super(BertClass, self).__init__()
        self.bert = model
        self.classifier = torch.nn.Linear(768, output_dim)

    def forward(self, 
            input_ids, 
            attention_mask,
        ):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        output = self.classifier(pooler)

        return output