from transformers import TFBertForMaskedLM, BertTokenizer, TFDistilBertForMaskedLM, DistilBertTokenizer, TFAlbertForMaskedLM, AlbertTokenizer
from extract_model_importance.extract_attention_ablations import extract_attention
from extract_model_importance.extract_saliency import extract_relative_saliency
from extract_model_importance import tokenization_util
from extract_human_fixations import data_extractor_geco, data_extractor_zuco


from transformers import AutoTokenizer, TFAutoModelForMaskedLM
import torch

def extract_all_attention(model, tokenizer, sentences, outfile, layer = None, head = None):
    with open(outfile, "w") as attention_file:
        for i, sentence in enumerate(sentences):
            # print(progress
            if i%500 ==0:
                print(i, len(sentences))
            tokens, relative_attention = extract_attention(model, tokenizer, sentence, layer, head)

        # merge word pieces if necessary

            tokens, merged_attention = tokenization_util.merge_subwords(tokens, relative_attention)
            tokens, merged_attention = tokenization_util.merge_hyphens(tokens, merged_attention)
            attention_file.write(str(tokens) + "\t" + str(merged_attention) + "\n")


def extract_all_saliency(model, embeddings, tokenizer, sentences, outfile):
    with open(outfile, "w") as saliency_file:
        for i, sentence in enumerate(sentences):
            # print(progress
            if i % 500 == 0:
                print(i, len(sentences))
            tokens, saliency = extract_relative_saliency(model, embeddings, tokenizer, sentence)

            # merge word pieces if necessary
            tokens, saliency = tokenization_util.merge_subwords(tokens, saliency)
            tokens, saliency = tokenization_util.merge_hyphens(tokens, saliency)
            saliency_file.write(str(tokens) + "\t" + str(saliency) + "\n")

def extract_all_human_importance(corpus):
    if corpus == "geco":
        data_extractor_geco()
    if corpus == "zuco":
        data_extractor_zuco


#corpora = ["geco", "zuco"]
corpora = ["geco"]
#models = ["distil", "albert","bert"]
models = ["distil"]


for corpus in corpora:
    # We skip extraction of human importance here because it takes quite long.
    #extract_all_human_importance(corpus)
    with open("results_attention/" + corpus + "_sentences.txt", "r") as f:
        sentences = f.read().splitlines()
    print("Processing Corpus: " + corpus)

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


        #outfile = "results_reproduced/" + corpus + "_" + modelname + "__reproduced_"
        outfile = "results_attention/" + corpus + "_" + modelname + "_"

        # print("Extracting attention for " + modelname)
        #for layer, _  in enumerate(model.layers):
        num_layers = 12
        if model == "distil":
            num_layers = 6
        if model == "tinybert": 
            num_layers = 2
        
        num_heads = 12
        if model == "tinybert":
            num_heads = 2
            
        for layer in range (num_layers):
            extract_all_attention(model, tokenizer, sentences, outfile+ "layer_" + str(layer) + "_attention.txt", layer = layer)
        
        for head in range(num_heads):
            extract_all_attention(model, tokenizer, sentences, outfile+ "head_" + str(head) + "_attention.txt", head = head)

        # Note: Saliency calculation takes much longer than attention calculation.
        # print("Extracting saliency for " + modelname)
        # extract_all_saliency(model,  embeddings, tokenizer, sentences, outfile + "saliency.txt")