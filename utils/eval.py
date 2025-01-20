import warnings
warnings.filterwarnings('ignore')

from bert_score import BERTScorer
import numpy as np
import jieba
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
from gensim.models import Word2Vec
from .generate_gpt import load_data


def eval_bertscore(hyps, gths):
    scorer = BERTScorer(lang='zh',model_type="bert-base-chinese",model_path="/Users/yiyi/workspace/models/bert-base-chinese")
    p,r,f = scorer.score(hyps, gths)
    result = dict()
    result['bertscore'] = {'p': np.mean(np.array(p)),
                           'r': np.mean(np.array(r)),
                           'f': np.mean(np.array(f))}
    return result


def eval_bleu(hyps, gths):
    tokens_hyps = [' '.join(jieba.cut(i)) for i in hyps]
    tokens_gths = [' '.join(jieba.cut(i)) for i in gths]
    result = {'bleu-1':0, 'bleu-2':0, 'bleu-3':0, 'bleu-4':0}
    smooth = SmoothingFunction().method1
    for hyp, gth in zip(tokens_hyps, tokens_gths):
        score1 = sentence_bleu([gth.split(' ')], (hyp.split(' ')), weights=(1, 0, 0, 0),smoothing_function=smooth)
        score2 = sentence_bleu([gth.split(' ')], (hyp.split(' ')), weights=(0, 1, 0, 0),smoothing_function=smooth)
        score3 = sentence_bleu([gth.split(' ')], (hyp.split(' ')), weights=(0, 0, 1, 0),smoothing_function=smooth)
        score4 = sentence_bleu([gth.split(' ')], (hyp.split(' ')), weights=(0, 0, 0, 1),smoothing_function=smooth)
        result['bleu-1'] += score1
        result['bleu-2'] += score2
        result['bleu-3'] += score3
        result['bleu-4'] += score4
    result['bleu-1'] /= len(tokens_hyps)
    result['bleu-2'] /= len(tokens_hyps)
    result['bleu-3'] /= len(tokens_hyps)
    result['bleu-4'] /= len(tokens_hyps)
    return result


def eval_rouge(hyps, gths):
    rouge = Rouge()
    tokens_hyps = [' '.join(jieba.cut(i)) for i in hyps]
    tokens_gths = [' '.join(jieba.cut(i)) for i in gths]
    scores = rouge.get_scores(tokens_hyps, tokens_gths, avg=True)
    return scores ## a dict like {'rouge-1': {'r': 1.0, 'p': 1.0, 'f': 1.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 1.0, 'p': 1.0, 'f': 1.0}}


def eval_TnM(hyps, gths):
    assert len(hyps)==len(gths)

    ## load the word embedding model
    model = Word2Vec.load(r"./models/skip-gram/skipgram.model")
    ## read the medical terms (THUOCL)
    with open(r"./THUOCL_medical.txt", 'r') as f:
        term_base = f.readlines()
    term_base = [i.split("\t")[0] for i in term_base]
    ## read the stopwords
    with open("cn_stopwords.txt", 'r') as f:
        stopwords = f.readlines()
    stopwords = " ".join([i.strip() for i in stopwords])
    ## tokenization and term extraction
    term_hyps = [" ".join([i for i in list(jieba.cut(text.strip())) if i in term_base and i not in stopwords]) for text in hyps]
    term_gths = [" ".join([i for i in list(jieba.cut(text.strip())) if i in term_base and i not in stopwords]) for text in gths]

    assert len(term_hyps)==len(term_gths)

    tp_list = []
    fp_list = []
    fn_list = []
    for hyp, gth in zip(term_hyps, term_gths):
        if hyp=="" or gth=="":
            print("No terms are extracted.")
            continue
        tp = 0
        fp = 0
        fn = 0
        topn_gth = " ".join([" ".join([j[0] for j in model.wv.most_similar(i, topn=3)]) for i in gth if i in model.wv.key_to_index])
        gth += " " + topn_gth
            
        for term in hyp:
            if term in gth:
                tp += 1
            else:
                fp += 1
        if tp==0:
            fn += 1
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

    
    P = sum(tp_list) / (sum(tp_list) + sum(fp_list))
    R = sum(tp_list) / (sum(tp_list) + sum(fn_list))
    micro_f1 = 2*P*R/(P+R)
    
    result = {'T3M':{'p':P, 'r':R, 'f':micro_f1}}
    return result

def eval_main(hyp_path, data_gth):
    with open(hyp_path, 'r',encoding='utf-8') as f:
        data_hyp = json.load(f)

    # idx = list(data_gth['Index'])
    # dignosis = list(data_gth['diagnosis'])
    # dbx_gths = list(data_gth['diagnosis_basis_ch'])
    # ddx_gths = list(data_gth['differential_diagnosis_ch'])

    hyps = list()
    gths = list()

    for id, value in data_hyp.items():
        gth = data_gth[data_gth.Index==float(id)]
        # dbx_gth = gth['diagnosis_basis_ch'].values[0]
        diagnosis_gth = gth['diagnosis'].values[0]
        ddx_gth = gth['differential_diagnosis_ch'].values[0]
    
        hyp = value['json']
        if list(hyp.keys())[0] in diagnosis_gth and len(hyp)>1:
            try:
                ddx_hyp = '\n'.join([':'.join(k_v) for i,k_v in enumerate(hyp.items()) if i>0])
            except:
                ddx_hyp = '\n'.join([':'.join((k, json.dumps(v,ensure_ascii=False))) for i,(k,v) in enumerate(hyp.items()) if i>0])
        elif list(hyp.keys())[0] not in diagnosis_gth:
            ddx_hyp = json.dumps(hyp,ensure_ascii=False)
        else:
            continue
        # dbx_hyp = ':'.join([list(hyp.keys())[0], list(hyp.values())[0]])                      ## the first one is the predicted diagnosis
        # ddx_hyp = '\n'.join([':'.join(k_v) for i,k_v in enumerate(hyp.items()) if i>0])     ## others are differential diagnosis
        
        gths.append(ddx_gth)
        hyps.append(ddx_hyp)




    tnm = eval_TnM(hyps, gths)
    rouge = eval_rouge(hyps, gths)
    bleu = eval_bleu(hyps, gths)
    bertscore = eval_bertscore(hyps, gths)
    eval_dict = {**tnm, **rouge, **bleu, **bertscore}
    # eval_dict = json.dumps(eval_dict, indent=2, ensure_ascii=False)
    print(hyp_path, eval_dict)
