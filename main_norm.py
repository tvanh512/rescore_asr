import os
import pandas as pd

import numpy as np
import whisper
import glob
import jiwer
from whisper.normalizers import EnglishTextNormalizer
from jiwer import wer
import torchaudio
import torch
from datasets import DatasetDict , Dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import DataLoader
import evaluate
import gc
from tqdm import tqdm
from ast import literal_eval

import torch.nn.functional as F
import time
from models import load_whisper, load_lm
from utils import whisper_english_normalizer, wer_wrap, save_df, find_oracle_pred, save_text, cal_wer
import yaml
import argparse
from rescore import rescore_ASR
from dataset import load_dataset

def decode(wavs_file,
           model, 
           processor, 
           tokenizer, 
           beam_size, 
           max_new_tokens=112,
           repetition_penalty=1,
           do_sample=False,
           no_repeat_ngram_size=None):
    start_time = time.time()
    preds, preds_beam, preds_score, preds_beam_score = [], [], [], []
    for i in range(len(wavs_file)):
        if i % 200 == 0:
            print(f"step {i}, running time: {time.time() - start_time}")
        # load the speech
        waveform, sample_rate = torchaudio.load(wavs_file[i])
        input_features = processor(waveform.squeeze(), 
                                   sampling_rate=sample_rate,
                                   return_tensors="pt").input_features 
        # generate token ids
        with torch.no_grad():
            out = (
                    model.generate(
                        input_features=input_features.to("cuda"),
                        return_dict_in_generate=True, 
                        output_scores=True, 
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        num_beams=1,
                        do_sample=do_sample
                    )
                )
            generated_tokens = out.sequences
            transition_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            pred_score = torch.mean(transition_scores,axis=1)
            #print('pred_score: ',pred_score)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            preds.extend(decoded_preds)
            preds_score.extend(pred_score.tolist())
            
            out_beam = (
                        model.generate(
                            input_features=input_features.to("cuda"),
                            temperature=0,
                            num_beams=beam_size,
                            num_return_sequences=beam_size,
                            return_dict_in_generate=True, 
                            output_scores=True,
                            max_new_tokens=max_new_tokens,
                            repetition_penalty=repetition_penalty,
                            do_sample=do_sample
                        )
                    )
            generated_tokens_beam = out_beam.sequences
            pred_beam_score = out_beam.sequences_scores
            pred_beam_score =  pred_beam_score.reshape(len(pred_score),-1)
            decoded_preds_beam = tokenizer.batch_decode(generated_tokens_beam, skip_special_tokens=True)
            decoded_preds_beam =  np.reshape(decoded_preds_beam,(len(decoded_preds),beam_size))
            preds_beam.extend(decoded_preds_beam.tolist())
            preds_beam_score.extend(pred_beam_score.tolist())
    return preds, preds_score, preds_beam, preds_beam_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from the specified YAML file
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Input
    dataset_name = config['dataset_name']
    dataset_config = config['dataset_config']

    # Output
    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    # ASR configuration
    asr_model_name_or_path = config['asr_model_name_or_path']
    beam_size = config['beam_size']
    no_repeat_ngram_size = config['no_repeat_ngram_size']
    # LM configuration
    lm = config['lm']
    lm_model_name_or_path = config['lm_model_name_or_path']

    # Load ASR model
    model, tokenizer,processor, _, _ = load_whisper(asr_model_name_or_path)

    # Load text normalizer
    normalizer = whisper_english_normalizer()

    # Load dataset:
    labels, wavs_file, labels_file = load_dataset(dataset_name, **dataset_config)

    # decode
    preds, preds_score, preds_beam, preds_beam_score = decode(wavs_file,
                                                              model,
                                                              processor,
                                                              tokenizer,
                                                              beam_size,
                                                              no_repeat_ngram_size)
    

    # calculate wer and save
    labels_norm_filter, preds_norm_filter, preds_beam_norm_filter,wavs_file_filter, labels_file_filter,_,_ = wer_wrap(labels,
                                                                                                                      preds,
                                                                                                                      preds_beam,
                                                                                                                      normalizer,
                                                                                                                      os.path.join(out_dir,'wer.txt'),
                                                                                                                      os.path.join(out_dir,'cer.txt'),
                                                                                                                      wavs_file,
                                                                                                                      labels_file)

    # rescore
    model, tokenizer = load_lm(lm, lm_model_name_or_path)
    preds_rescore_norm_filter, prob_sequences = rescore_ASR(preds_beam_norm_filter,preds_norm_filter,model, tokenizer)

    data_dict = {
        'labels': labels,
        'preds': preds,
        'preds_beam': preds_beam,
        'preds_score': preds_score,
        'preds_beam_score': preds_beam_score,
        'wavs_file': wavs_file,
        'labels_file': labels_file,
        'prob_sequences': prob_sequences
    }
    save_df(os.path.join(out_dir, 'preds_score.csv'), data_dict)

    # find the oracle predictions, to get an bounded wer in case we have perfect ranking
    
    oracle_pred_norm_filter = find_oracle_pred(labels_norm_filter, preds_norm_filter, preds_beam_norm_filter)
    _, wer_align_oracle,_, cer_align_oracle = cal_wer(labels_norm_filter, oracle_pred_norm_filter)
    
    save_text(os.path.join(out_dir,'wer_oracle.txt'), wer_align_oracle)
    save_text(os.path.join(out_dir,'cer_oracle.txt'), cer_align_oracle)
    data_dict = {
        'labels_norm_filter': labels_norm_filter,
        'preds_norm_filter': preds_norm_filter,
        'preds_beam_norm_filter': preds_beam_norm_filter, 
        'oracle_pred_norm_filter': oracle_pred_norm_filter,
        'preds_rescore_norm_filter': preds_rescore_norm_filter,
        'wavs_file_filter': wavs_file_filter,
        'labels_file_filter': labels_file_filter
    }
    save_df(os.path.join(out_dir, 'oracle.csv'), data_dict)

if __name__ == "__main__":
    main()
