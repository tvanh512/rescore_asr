import pandas as pd
import yaml
import argparse
from utils import cal_speech_len, save_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from the specified YAML file
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Input
    preds_score_csv = config['preds_score_csv']
    # Threshold to distinguish speech from silence
    threshold = config['threshold']

    # Output
    speech_len_csv = config['speech_len_csv']

    preds_score_df = pd.read_csv(preds_score_csv)
    wavs_file = preds_score_df['wavs_file'].to_list()

    speechs_len = []
    wavs_len = []
    for i in range(len(wavs_file)):
        wav_len, speech_len = cal_speech_len(wavs_file[i], threshold =threshold)
        wavs_len.append(wav_len)
        speechs_len.append(speech_len)
    data_dict = {
        'wavs_file': wavs_file,
        'wavs_len': wavs_len, 
        'speechs_len': speechs_len}
    save_df(speech_len_csv, data_dict)

if __name__ == "__main__":
    main()
