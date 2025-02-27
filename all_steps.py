import yaml
import decode
import rescore
import argparse
import config_utils
from config_utils
def run_decode(config):
    decode.run(config)

def run_rescore(config):
    rescore.run(config)

def run_split(run_steps, decode_cfg, rescore_cfg):
    if 'decode' in run_steps:
        run_decode(decode_cfg)
    if 'rescore_params_search' in run_steps:
        rescore_cfg['hyp_search'] = True
        run_rescore(rescore_cfg)
    if 'rescore' in run_steps:
        rescore_cfg['hyp_search'] = False
        run_rescore(rescore_cfg)

def run():
    run_split()

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--config", help="Path to the YAML configuration file")
    #args = parser.parse_args()

    # Load configuration from the specified YAML file
    #with open(args.config, 'r') as yaml_file:
    #    config = yaml.safe_load(yaml_file)
    #run(config)
    
    # Fine the best 
    cfg_asr = get_decode_cfg_asr()
    cfg_lm = get_lm_cfg()
    cfg_rescore_path
    run(run_steps, decode_cfg, rescore_cfg)