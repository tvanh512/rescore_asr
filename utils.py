import jiwer
import torch
from whisper.normalizers import EnglishTextNormalizer
import numpy as np
import pandas as pd
from ast import literal_eval
import librosa
from ax.service.ax_client import AxClient
import yaml
import os
from wer_utils import format_text_for_wer

# Visualize the error rate
def cal_wer(labels, preds):
    jiwer_wer_out = jiwer.process_words(labels, preds)
    jiwer_cer_out = jiwer.process_characters(labels, preds)
    wer_align = jiwer.visualize_alignment(jiwer_wer_out,skip_correct=False)
    cer_align = jiwer.visualize_alignment(jiwer_cer_out, skip_correct=False)
    return jiwer_wer_out.wer, wer_align,jiwer_cer_out.cer, cer_align

def save_text(out_file, content):
    with open(out_file, "w") as f:
        f.write(content)

def save_list(out_file, data_list):
    with open(out_file, mode='w') as f:
        for item in data_list:
            f.write(item + "\n")



# return text normalizer 
def whisper_english_normalizer():
    normalizer = EnglishTextNormalizer()
    # Whisper normalizer will remove hmm|mm|mhm|mmm|uh|um but we want to keep it
    # https://github.com/openai/whisper/blob/248b6cb124225dd263bb9bd32d060b6517e067f8/whisper/normalizers/english.py
    normalizer.ignore_patterns = ""
    return normalizer

def norm_and_filter(labels,preds,preds_beam, normalizer, wavs_file):
    labels_norm = [ normalizer(str(text)) for text in labels]
    preds_norm = [ normalizer(str(text)) for text in preds]
    preds_beam_norm = []
    for i in range(len(preds_beam)):
        preds_beam_norm.append([normalizer(str(text)) for text in preds_beam[i]])
    labels_norm_filter, preds_norm_filter, preds_beam_norm_filter = [], [], []
    wavs_file_filter = []
    for i in range(len(labels_norm)):
        if labels_norm[i] != "":
            labels_norm_filter.append(labels_norm[i])
            preds_norm_filter.append(preds_norm[i])
            preds_beam_norm_filter.append(preds_beam_norm[i])
            wavs_file_filter.append(wavs_file[i])
    return labels_norm_filter, preds_norm_filter, preds_beam_norm_filter, wavs_file_filter

def wer_wrap(labels, preds, preds_beam, normalizer, wer_file, cer_file,wavs_file):
    # wrap normalization, calculate wer and save wer in one step
    labels_norm_filter, preds_norm_filter, preds_beam_norm_filter, wavs_file_filter= norm_and_filter(labels, preds, preds_beam, normalizer,  wavs_file)
    wer, wer_align,cer, cer_align = cal_wer(labels_norm_filter, preds_norm_filter)
    save_text(wer_file, wer_align)
    save_text(cer_file, cer_align)
    return labels_norm_filter, preds_norm_filter, preds_beam_norm_filter, wavs_file_filter, wer, cer 

def save_df(out_file, data_dict):
    df = pd.DataFrame(data_dict)
    df.to_csv(out_file,index=False)

def find_oracle_pred(labels,preds, preds_beam):
    oracle_preds = []
    oracle_ids =[]
    for i in range(len(labels)):
        all_preds  = [preds[i]] + preds_beam[i] 
        wer_scores = [jiwer.wer(labels[i], all_preds[j]) if str(all_preds[j]).strip() else 100000 for j in range(len(all_preds))]
        best_ids = int(np.argmin(wer_scores))
        oracle_preds.append(all_preds[best_ids])
        oracle_ids.append(best_ids)
    return oracle_preds, oracle_ids

def get_normalizer(text_norm):
    if text_norm == 'whisper':
        normalizer = whisper_english_normalizer()
    else:
        normalizer = format_text_for_wer
    return normalizer

# Find the sentence log probability from a language model
def logprobs(model, tokenizer, input_texts,device='cuda'):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    bos_token_id = tokenizer.bos_token_id
    
    # Add beginning of sentence bos for every sentence in the batch
    bos_token_id = torch.tensor([[bos_token_id]] * len(input_ids))
    bos_token_id = bos_token_id.to(device)
    input_ids = torch.cat((bos_token_id, input_ids), dim=1)

    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1)

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    prob_tokens = []
    prob_sequences =[]
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        prob_sequence = 0
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
                prob_sequence += p.item()
        prob_sequences.append(prob_sequence)
        prob_tokens.append(text_sequence)
    return prob_sequences

def rescore_ASR(preds_beam,preds,model, tokenizer):
    preds_rescore = []
    prob_sequences = []
    for i in range(len(preds_beam)):
        candidates =  [preds[i]] + preds_beam[i]
        prob_sequence = [logprobs(model, tokenizer, candidate)[0] for candidate in candidates if str(candidate).strip()]
        best_candidate = candidates[int(np.argmax(prob_sequence))]
        preds_rescore.append(best_candidate)
        prob_sequences.append(prob_sequence)
    return preds_rescore, prob_sequences

def str2list(st):
    st = [literal_eval(s) for s in st]
    return st

def get_decode_result(decode_csv):
    preds_score_df = pd.read_csv(decode_csv)
    wavs_file = preds_score_df['wavs_file'].to_list()
    labels = preds_score_df['labels'].to_list()
    preds = preds_score_df['preds'].to_list()
    preds_score = preds_score_df['preds_score'].to_list()
    preds_beam = str2list(preds_score_df['preds_beam'].to_list())
    preds_beam_score = str2list(preds_score_df['preds_beam_score'].to_list())
    wavs_len = preds_score_df['wavs_len'].to_list()
    return wavs_file, labels, preds, preds_score, preds_beam, preds_beam_score, wavs_len

def get_lm_score(lm_csv):
    lm_df = pd.read_csv(lm_csv)
    lm_scores = lm_df['lm_scores'].to_list()
    lm_beam_scores = str2list(lm_df['lm_beam_scores'].to_list())
    return lm_scores, lm_beam_scores

def get_speech_len(speech_len_csv):
    if speech_len_csv:
        df = pd.read_csv(speech_len_csv)
        speechs_len = df['speechs_len'].to_list()
        wavs_len = df['wavs_len'].to_list()
        return wavs_len, speechs_len
    else:
        return None
    
def cal_speech_len(wav_file, threshold = 0.005, limit = 30):
    y, sr = librosa.load(wav_file)
    
    # Only calculate for the first 'limit' second
    if len(y) > limit * sr:
        new_len = limit * sr
        y = y[:new_len]
        
    wav_len = len(y)/sr         
    # Calculate energy envelope
    energy = librosa.feature.rms(y=y)[0]
    # Set a threshold to distinguish speech from silence
    threshold = threshold
    # Identify speech regions using energy threshold
    speech_regions = [i for i, e in enumerate(energy) if e > threshold]
    speech_len = wav_len * len(speech_regions)/len(energy)
    return wav_len, speech_len

def get_ax_result(ax_json):
    ax_client = (AxClient.load_from_json_file(filepath=ax_json)) 
    #ax_client.get_trials_data_frame()
    best_parameters, lowest_wer = ax_client.get_best_parameters()
    return best_parameters, lowest_wer

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as f:
            # Load the data from YAML file
            data = yaml.safe_load(f)
            return data
    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error: {e}")
    return None


def get_rescore_dir_prefix(decode_dir, search_space,equation, lm_id):
    if search_space:
        search_space = 'search_space_' + str(search_space) 
    else:
        search_space =''
    rescore_dir_prefix = os.path.join(decode_dir, 'rescore','equation_' + str(equation), lm_id, str(search_space))
    return rescore_dir_prefix

def get_rescore_dir_name(decode_dir, search_space,equation, lm_id, **kwargs):
    rescore_dir_prefix = get_rescore_dir_prefix(decode_dir, search_space,equation,lm_id)
    if equation in [6]:
        rescore_dir  = os.path.join(rescore_dir_prefix, 'asr_' + str(round(kwargs['asr_weight'],5)) + 'lm_' + str(round(kwargs['lm_weight'],5))+ 'short_' + str(round(kwargs['short_penalty_weight'],5)))
    return rescore_dir

def get_subset_name(csv_path, threshold):
    df_subset_csv = csv_path.replace('.csv', '_subset_threshold_' + str(threshold) + '.csv')
    return df_subset_csv

def get_subset_helper(csv_path,threshold, subset_field, column ='wavs_file',):
    df = pd.read_csv(csv_path)
    df_subset = df[df[column].isin(subset_field)]
    df_subset_csv = get_subset_name(csv_path, threshold)
    df_subset.to_csv(df_subset_csv,index=False)
    return df_subset_csv

def get_subset(decode_dir, preds_score_csv, lm_csv, threshold=0.3):
    preds_score_csv_subset = get_subset_name(preds_score_csv, threshold)
    lm_csv_subset = get_subset_name(lm_csv, threshold)
    if not os.path.exists(preds_score_csv_subset):
        oracle_df = pd.read_csv(os.path.join(decode_dir,'oracle.csv'))
        labels_norm_filter = oracle_df['labels_norm_filter'].to_list()
        oracle_pred_norm_filter= oracle_df['oracle_pred_norm_filter'].to_list()
        wavs_file_filter = oracle_df['wavs_file_filter']
        # This function filters wav files that has lower WERs
        # For the purpose of hyperparameter search
        wavs_file_subset, wer_dist= [], []
        for i in range(len(labels_norm_filter)):
            wer_single = jiwer.wer(labels_norm_filter[i],oracle_pred_norm_filter[i] )
            wer_dist.append(wer_single)
            if wer_single < threshold:
                wavs_file_subset.append(wavs_file_filter[i])
        preds_score_csv_subset = get_subset_helper(preds_score_csv,threshold,wavs_file_subset)
    if not os.path.exists(lm_csv_subset):
        df = pd.read_csv(preds_score_csv_subset)
        wavs_file_subset = df['wavs_file']
        lm_csv_subset = get_subset_helper(lm_csv,threshold, wavs_file_subset)
    return preds_score_csv_subset, lm_csv_subset

def get_config(cfg, split):
    # Get decode config
    decode_dataset_cfg = read_yaml(os.path.join(cfg['root_cfg'], cfg[split]['decode_dataset_cfg']))
    decode_dir = decode_dataset_cfg['out_dir']
    decode_asr_cfg = read_yaml(os.path.join(cfg['root_cfg'],cfg['decode_asr_cfg']))
    decode_cfg = {**decode_asr_cfg, **decode_dataset_cfg}
    preds_score_csv = os.path.join(decode_dir,'preds_score.csv')
    # Get LM score config.
    lm_score_cfg = read_yaml(os.path.join(cfg['root_cfg'],cfg['lm_cfg']))
    # lm score output file
    lm_id = lm_score_cfg['lm_id']
    lm_csv = os.path.join(decode_dir, lm_id + '.csv')
    lm_score_cfg['preds_score_csv'] = preds_score_csv
    lm_score_cfg['lm_csv'] = lm_csv
    # Get rescore config
    equation = cfg['equation']
    rescore_cfg = read_yaml(os.path.join(cfg['root_cfg'],cfg[split]['rescore_cfg']))
    print("os.path.join(cfg['root_cfg'],cfg[split]['rescore_cfg'])",os.path.join(cfg['root_cfg'],cfg[split]['rescore_cfg']))
    rescore_cfg['decode_dir'] = decode_dir
    rescore_cfg['text_norm'] = decode_asr_cfg['text_norm']
    rescore_cfg['equation'] = equation
    rescore_cfg ={**rescore_cfg,**lm_score_cfg}
    if rescore_cfg['hyp_search']:
        rescore_dir_prefix = get_rescore_dir_prefix(decode_dir, rescore_cfg['search_space'],equation,lm_id)
        # path to save the ax hyper-parameters search result
        ax_json = os.path.join(rescore_dir_prefix,'ax.json')   
        ax_exp_name =  decode_dataset_cfg['dataset_name'] + split + lm_id+ '_equation_'+ str(equation) + '_ss_' +  str(rescore_cfg['search_space'])                                   
        rescore_cfg['ax_json']= ax_json
        rescore_cfg['ax_exp_name']= ax_exp_name
    return decode_cfg, lm_score_cfg, rescore_cfg

