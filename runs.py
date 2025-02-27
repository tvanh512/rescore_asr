import yaml
import decode
import rescore
import lm_score
import argparse
from utils import get_config, get_subset
import os

def run_decode(config):
    decode.run(config)

def run_lm_score(config):
    lm_score.run(config)
    
def run_rescore(config):
    rescore.run(config)

def run(config):
    ax_json = None 
    print('config:', config)
    for k, v in config.items():
        if k in ['train', 'dev', 'test','test_corrected']:
            split, cfg = k,v
            print('split', split)
            print('cfg', cfg)
            if not cfg['skip']:
                decode_cfg, lm_score_cfg, rescore_cfg = get_config(config, split)
                if cfg['decode']:
                    run_decode(decode_cfg)
                if cfg['lm_score']:
                    run_lm_score(lm_score_cfg)
                if 'subset' in rescore_cfg and rescore_cfg['subset']:
                    preds_score_csv_subset, lm_csv_subset = get_subset(rescore_cfg['decode_dir'], rescore_cfg['preds_score_csv'], rescore_cfg['lm_csv'], rescore_cfg['wer_threshold_subset'])
                    rescore_cfg['preds_score_csv'] = preds_score_csv_subset
                    rescore_cfg['lm_csv'] = lm_csv_subset
                print('decode_cfg:',decode_cfg)
                print('lm_score_cfg:',lm_score_cfg)
                print('rescore_cfg:', rescore_cfg)
                if cfg['rescore']:
                    if rescore_cfg['hyp_search']:
                        #load ax_json from train or dev
                        ax_json = rescore_cfg['ax_json'] 
                    # There are 3 ways to load the best params for testing: automatically from the dev or training set, 
                    # specifying 'ax_json' in the rescore config manually or entering the parameters manually.
                    if 'load_params_from_search' in rescore_cfg and rescore_cfg['load_params_from_search'] and not rescore_cfg['ax_json']:
                        rescore_cfg['ax_json'] = ax_json
                    run_rescore(rescore_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the YAML configuration file")
    args = parser.parse_args()
    #Load configuration from the specified YAML file
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    run(config)
            
            
