from transformers import TFBertForMaskedLM, BertTokenizer, TFDistilBertForMaskedLM, DistilBertTokenizer, TFAlbertForMaskedLM, AlbertTokenizer
from extract_model_importance.extract_attention_ablations import extract_attention
from extract_model_importance.extract_saliency import extract_relative_saliency
from extract_model_importance import tokenization_util
from extract_human_fixations import data_extractor_geco, data_extractor_zuco


from transformers import AutoTokenizer, TFAutoModelForMaskedLM
import torch

#corpora = ["geco", "zuco"]
corpora = ["zuco"]
models = ["distil", "albert","bert","tinybert","minilm"]


for modelname in models:
    # TODO: this could be moved to a config file
    if modelname == "bert":
        MODEL_NAME = 'bert-base-uncased'
        model = TFBertForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        embeddings = model.bert.embeddings.word_embeddings
    if modelname == "albert":
        MODEL_NAME = 'albert-base-v2'
        model = TFAlbertForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True)
        tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
        embeddings = model.albert.embeddings.word_embeddings
    if modelname == "distil":
        MODEL_NAME = 'distilbert-base-uncased'
        model = TFDistilBertForMaskedLM.from_pretrained(MODEL_NAME, output_attentions=True)
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        embeddings = model.distilbert.embeddings.word_embeddings

    if modelname == "tinybert":
        #MODEL_NAME = 'sentence-transformers/paraphrase-TinyBERT-L6-v2'
        MODEL_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2'
        model = TFAutoModelForMaskedLM.from_pretrained(MODEL_NAME,output_attentions=True, from_pt=True)
        tokenizer = AutoTokenizer.from_pretrained('./saved_tinybert/')

        # run this locally first
        # model = TFAutoModelForMaskedLM.from_pretrained(MODEL_NAME,output_attentions=True, from_pt=True)
        # model.save_pretrained('./saved_tinybert/')
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # tokenizer.save_pretrained('./saved_tinybert/')

        embeddings = model.bert.embeddings.word_embeddings # check this

    if modelname == 'minilm':
        MODEL_NAME = 'microsoft/MiniLM-L12-H384-uncased'
        #MODEL_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
        model = TFAutoModelForMaskedLM.from_pretrained(MODEL_NAME,output_attentions=True, from_pt=True)
        tokenizer = AutoTokenizer.from_pretrained('./saved_minilm/')

        # run this locally first
        # model = TFAutoModelForMaskedLM.from_pretrained(MODEL_NAME,output_attentions=True, from_pt=True)
        # model.save_pretrained('./saved_minilm/')
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # tokenizer.save_pretrained('./saved_minilm/')

        embeddings = model.bert.embeddings.word_embeddings
    print(model.layers)
