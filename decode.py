import os
import numpy as np
import torchaudio
import torch
import time
from models import load_whisper, load_whisper_finetune
from utils import get_normalizer, wer_wrap, save_df, find_oracle_pred, save_text, cal_wer
import yaml
import argparse
from dataset import load_dataset
from wer_utils import format_text_for_wer

def decode(wavs_file,
           model, 
           processor, 
           tokenizer, 
           asr_generate_greedy_cfg,
           asr_generate_beam_cfg):
    start_time = time.time()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    preds, preds_beam, preds_score, preds_beam_score, transitions_score, wavs_len = [], [], [], [], [], []
    for i in range(len(wavs_file)):
        if i % 200 == 0:
            print(f"step {i}, running time: {time.time() - start_time}")
        # load the speech
        waveform, sample_rate = torchaudio.load(wavs_file[i])
        # derive wav_len in second
        wavs_len.append(waveform.shape[1] / sample_rate)
        input_features = processor(waveform.squeeze(), 
                                   sampling_rate=sample_rate,
                                   return_tensors="pt").input_features 
        # generate token ids
        with torch.no_grad():
            out = (
                    model.generate(
                        input_features=input_features.to("cuda"),
                        temperature=0,
                        return_dict_in_generate=True, 
                        output_scores=True, 
                        num_beams=1,
                        forced_decoder_ids=forced_decoder_ids,
                        **asr_generate_greedy_cfg
                    )
                )
            generated_tokens = out.sequences
            transition_score = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            transitions_score.append(transition_score.detach().cpu().tolist())
            pred_score = torch.mean(transition_score,axis=1)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            preds.extend(decoded_preds)
            preds_score.extend(pred_score.tolist())
            
            out_beam = (
                        model.generate(
                            input_features=input_features.to("cuda"),
                            temperature=0,
                            return_dict_in_generate=True, 
                            output_scores=True,
                            forced_decoder_ids=forced_decoder_ids,
                            **asr_generate_beam_cfg
                        )
                    )
            generated_tokens_beam = out_beam.sequences
            pred_beam_score = out_beam.sequences_scores
            pred_beam_score =  pred_beam_score.reshape(len(pred_score),-1)
            decoded_preds_beam = tokenizer.batch_decode(generated_tokens_beam, skip_special_tokens=True)
            decoded_preds_beam = np.reshape(decoded_preds_beam,(len(decoded_preds),asr_generate_beam_cfg['num_beams']))
            preds_beam.extend(decoded_preds_beam.tolist())
            preds_beam_score.extend(pred_beam_score.tolist())
    return preds, preds_score, preds_beam, preds_beam_score, transitions_score, wavs_len

def run(config):
    # Input
    dataset_name = config['dataset_name']
    dataset_config = config['dataset_config']

    # Output
    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    # ASR configuration
    asr_model_name_or_path = config['asr_model_name_or_path']
    text_norm = config['text_norm']
    asr_generate_greedy_cfg = config['asr_generate_greedy_cfg']
    asr_generate_beam_cfg = config['asr_generate_beam_cfg']
    

    # Load ASR model
    if asr_model_name_or_path in ['openai/whisper-large-v2']:
        model, tokenizer,processor, _, _ = load_whisper(asr_model_name_or_path)
    else:
        print('load custom model')
        model, tokenizer,processor, _, _ = load_whisper_finetune(asr_model_name_or_path)
    # Load text normalizer
    normalizer = get_normalizer(text_norm)

    # Load dataset:
    labels, wavs_file, extra = load_dataset(dataset_name, **dataset_config)

    # decode
    preds, preds_score, preds_beam, preds_beam_score, transitions_score, wavs_len = decode(wavs_file,
                                                                                model, 
                                                                                processor, 
                                                                                tokenizer, 
                                                                                asr_generate_greedy_cfg,
                                                                                asr_generate_beam_cfg)
    

    # calculate wer and save
    labels_norm_filter, preds_norm_filter, preds_beam_norm_filter,wavs_file_filter,_,_ = wer_wrap(labels,
                                                                                                                      preds,
                                                                                                                      preds_beam,
                                                                                                                      normalizer,
                                                                                                                      os.path.join(out_dir,'wer.txt'),
                                                                                                                      os.path.join(out_dir,'cer.txt'),
                                                                                                                      wavs_file) 
    data_dict = {
        'labels': labels,
        'preds': preds,
        'preds_beam': preds_beam,
        'preds_score': preds_score,
        'preds_beam_score': preds_beam_score,
        'wavs_file': wavs_file,
        'transitions_score': transitions_score,
        'wavs_len': wavs_len
    }
    if extra:
        data_dict = dict(data_dict, **extra)
    save_df(os.path.join(out_dir, 'preds_score.csv'), data_dict)

    # find the oracle predictions, to get an bounded wer in case we have perfect ranking
    
    oracle_pred_norm_filter, oracle_ids = find_oracle_pred(labels_norm_filter, preds_norm_filter, preds_beam_norm_filter)
    _, wer_align_oracle,_, cer_align_oracle = cal_wer(labels_norm_filter, oracle_pred_norm_filter)
    
    save_text(os.path.join(out_dir,'wer_oracle.txt'), wer_align_oracle)
    save_text(os.path.join(out_dir,'cer_oracle.txt'), cer_align_oracle)
    data_dict = {
        'labels_norm_filter': labels_norm_filter,
        'preds_norm_filter': preds_norm_filter,
        'preds_beam_norm_filter': preds_beam_norm_filter, 
        'oracle_pred_norm_filter': oracle_pred_norm_filter,
        'oracle_ids': oracle_ids, 
        'wavs_file_filter': wavs_file_filter
    }
    save_df(os.path.join(out_dir, 'oracle.csv'), data_dict)

if __name__ == "__main__":
    print('run decode lang en')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from the specified YAML file
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    run(config)
