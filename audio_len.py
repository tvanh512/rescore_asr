import pandas as pd
import yaml
import argparse
from utils import cal_audio_len, save_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from the specified YAML file
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Input
    preds_score_csv = config['preds_score_csv']

    # Output
    audios_len_csv = config['audios_len_csv']
    preds_score_df = pd.read_csv(preds_score_csv)
    wavs_file = preds_score_df['wavs_file'].to_list()

    audios_len = []
    for i in range(len(wavs_file)):
        audios_len.append(cal_audio_len(wavs_file[i]))
    
    data_dict = {
        'wavs_file': wavs_file,
        'audios_len': audios_len}
    save_df(audios_len_csv, data_dict)

if __name__ == "__main__":
    main()
