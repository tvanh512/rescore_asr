import pandas as pd
from utils import str2list, save_df
from models import load_lm
from rescore_utils import logprobs
import yaml
import argparse

def run(config):
    # LM configs
    lm_model_name_or_path = config['lm_model_name_or_path']
    lm = config['lm']
    preds_score_csv = config['preds_score_csv']
    lm_csv = config['lm_csv']

    # Load data
    preds_score_df = pd.read_csv(preds_score_csv)
    wavs_file = preds_score_df['wavs_file'].to_list()
    preds_beam = str2list(preds_score_df['preds_beam'].to_list())
    preds = preds_score_df['preds'].to_list()   

    # Load LM
    lm_model, lm_tokenizer = load_lm(lm, lm_model_name_or_path)

    # Start scoring
    lm_scores = []
    lm_beam_scores = []
    for i in range(len(preds)):
        lm_scores.append(logprobs(lm_model, lm_tokenizer, preds[i])[0] if str(preds[i]).strip() else -100000)
        lm_beam_scores.append([logprobs(lm_model, lm_tokenizer, pred)[0] if str(pred).strip() else -100000 for pred in preds_beam[i]])

    data_dict = {
        'wavs_file': wavs_file,
        'lm_scores': lm_scores, 
        'lm_beam_scores': lm_beam_scores}
    save_df(lm_csv, data_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from the specified YAML file
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    run(config)