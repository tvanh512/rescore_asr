from utils import get_normalizer, get_decode_result, get_lm_score, get_ax_result, get_rescore_dir_name
import yaml
import argparse
from wer_utils import format_text_for_wer
from rescore_utils import evaluate_wer, get_hyp_search_params, get_rescore_params
from ax.service.ax_client import AxClient, ObjectiveProperties
from transformers import AutoTokenizer, WhisperTokenizer
from models import load_tokenizer
import os

def train_evaluate(params,
                  preds_score_csv, 
                  lm,
                  lm_model_name_or_path,
                  lm_csv,
                  normalizer,
                  equation, 
                  rescore_dir,
                  asr_model_name_or_path,
                  language):
    print('parameters:', params )
    wavs_file, labels, preds, preds_score, preds_beam, preds_beam_score, wavs_len = get_decode_result(preds_score_csv)
    lm_scores, lm_beam_scores = get_lm_score(lm_csv)
    lm_tokenizer = load_tokenizer(lm,lm_model_name_or_path)
    asr_tokenizer = WhisperTokenizer.from_pretrained(asr_model_name_or_path, language=language)
    return evaluate_wer(preds,
                        preds_score,
                        lm_scores,
                        preds_beam, 
                        preds_beam_score,
                        lm_beam_scores, 
                        wavs_file,
                        wavs_len,
                        labels,
                        normalizer,
                        equation,
                        rescore_dir,
                        lm_tokenizer,
                        asr_tokenizer,
                        **params)[0]

def run(config):
    # Inputs configs, inputs are the decode's result
    preds_score_csv = config['preds_score_csv']
    decode_dir = config['decode_dir']
    lm_csv = config['lm_csv']
    # Rescore configs
    lm = config['lm']
    lm_id = config['lm_id']
    lm_model_name_or_path = config['lm_model_name_or_path']
    text_norm = config['text_norm']
    equation = config['equation']
    search_space=config['search_space']
    # Load text normalizer
    normalizer = get_normalizer(text_norm)
    hyp_search = config['hyp_search']
    asr_model_name_or_path = config['asr_model_name_or_path']
    language = config['language']
    
    # Hyperparameter search configs:
    
    if hyp_search:
        
        ax_exp_name = config['ax_exp_name']
        #lower_bound = config['lower_bound']
        #upper_bound = config['upper_bound']
        num_trials = config['num_trials']
        #log_scale = config['log_scale']
        ax_json = config['ax_json']
        search_space = config['search_space']
        # Adapted from https://ax.dev/tutorials/tune_cnn_service.html
        # Create a client object to interface with Ax APIs. By default this runs locally without storage.
        ax_client = AxClient()
        params = get_hyp_search_params(**config)
        ax_client.create_experiment(
            name=ax_exp_name,  # The name of the experiment.
            parameters=params,
            objectives={"wer": ObjectiveProperties(minimize=True)}, 
            overwrite_existing_experiment=True
        )

        for i in range(num_trials):
            params, trial_index = ax_client.get_next_trial()
            
            rescore_dir = get_rescore_dir_name(decode_dir, 
                                               search_space,
                                               equation,
                                               lm_id,
                                               **params)
            ax_client.complete_trial(trial_index=trial_index, raw_data = train_evaluate(params,
                                                                                        preds_score_csv, 
                                                                                        lm,
                                                                                        lm_model_name_or_path,
                                                                                        lm_csv,
                                                                                        normalizer,
                                                                                        equation, 
                                                                                        rescore_dir,
                                                                                        asr_model_name_or_path,
                                                                                        language
                                                                                        ))
        ax_client.save_to_json_file(ax_json) 
        best_parameters, best_wer_rescore = ax_client.get_best_parameters()
        print('best_parameters :', best_parameters)
        print('lowest WER: ', best_wer_rescore)

        print(ax_client.get_trials_data_frame())
    else:
        lm_tokenizer = load_tokenizer(lm,lm_model_name_or_path)
        asr_tokenizer = WhisperTokenizer.from_pretrained(asr_model_name_or_path, language=language)
        wavs_file, labels, preds, preds_score, preds_beam, preds_beam_score, wavs_len = get_decode_result(preds_score_csv)
        lm_scores, lm_beam_scores = get_lm_score(lm_csv)
        if config['load_params_from_search']:
            rescore_params,_ = get_ax_result(config['ax_json']) 
        else:
            rescore_params = get_rescore_params(**config)
        rescore_dir = get_rescore_dir_name(decode_dir, 
                                           search_space,
                                           equation,
                                           lm_id,
                                           **rescore_params)
        evaluate_wer(preds,
                    preds_score,
                    lm_scores,
                    preds_beam, 
                    preds_beam_score,
                    lm_beam_scores, 
                    wavs_file,
                    wavs_len,
                    labels,
                    normalizer,
                    equation,
                    rescore_dir,
                    lm_tokenizer,
                    asr_tokenizer,
                    **rescore_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from the specified YAML file
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    run(config)