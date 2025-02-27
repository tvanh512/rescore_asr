import os
from utils import read_yaml
from rescore_utils import get_rescore_dir_prefix

def get_decode_cfg_asr(asr_model_name_or_path = 'openai/whisper-large-v2',
                       beam_size = 16,
                       max_new_tokens = 112,
                       repetition_penalty = 1, 
                       do_sample =  False,
                       text_norm = 'custom'
                       ):
    # text_norm could be whisper or custom}
    cfg =   {'asr_model_name_or_path': asr_model_name_or_path,
            'beam_size': beam_size,
            'max_new_tokens': max_new_tokens,
            'repetition_penalty': repetition_penalty, 
            'do_sample': do_sample,
            'text_norm': text_norm  
            }
    return cfg

def get_decode_cfg_dataset(dataset_name,
                            data_dir,
                            meta_csv,
                            wavs_file_col,
                            labels_col, 
                            data_root,
                            out_dir,
                            num_samples = None,
                            ):
    dataset_config= {'data_dir' : data_dir,
                    'meta_csv' : meta_csv,
                    'num_samples': num_samples,
                    'wavs_file_col': wavs_file_col,
                    'labels_col': labels_col,
                    'data_root': data_root}
    cfg = {'dataset_name' : dataset_name,
           'dataset_config': dataset_config,
           'out_dir': out_dir}
    return cfg

def get_decode_cfg_dataset_levi_dev(dataset_name = 'levi_lofi_5',
                                    data_dir = '/home/vtrinh/datasets/LEVI/first11',
                                    meta_csv = '/home/vtrinh/datasets/metadata/LEVI_lofi5/LEVI_lofi5_student.csv',
                                    wavs_file_col = 'wavs_file',
                                    labels_col = 'utterance', 
                                    data_root = '$DATAROOT/first11',
                                    out_dir = '/home/vtrinh/misc/LEVI/LEVI_lofi5_student/whisper_large_v2/',
                                    num_samples = None, 
                                    ):
    return get_decode_cfg_dataset(dataset_name,
                            data_dir,
                            meta_csv,
                            wavs_file_col,
                            labels_col, 
                            data_root,
                            out_dir,
                            num_samples)

def get_lm_cfg(lm,lm_model_name_or_path):
    cfg =  {'lm':lm,
            'lm_model_name_or_path': lm_model_name_or_path}
    return cfg

def get_lm_score_cfg_dataset(decode_dir, lm_csv_name):
    # Input
    preds_score_csv = os.path.join(decode_dir, 'preds_score.csv')
    # Output
    lm_csv = os.path.join(decode_dir, lm_csv_name)
    cfg = {'preds_score_csv': preds_score_csv,
            'lm_csv': lm_csv
           }
    return cfg

def get_hyper_params_cfg(ax_json, 
                         ax_exp_name, 
                         search_space, 
                         num_trials = 400,
                         asr_lower_bound = 0,
                         asr_upper_bound = 1,
                         lm_lower_bound = 0,
                         lm_upper_bound = 1,
                         short_lower_bound = 0,
                         short_upper_bound = 1,
                         min_word = 1000,
                         min_prob = 1,
                         log_scale = False):

    cfg = { 'ax_json': ax_json,
            'ax_exp_name': ax_exp_name,
            'search_space': search_space,
            'num_trials': num_trials,
            'asr_lower_bound': asr_lower_bound,
            'asr_upper_bound': asr_upper_bound,
            'lm_lower_bound': lm_lower_bound,
            'lm_upper_bound': lm_upper_bound,
            'short_lower_bound': short_lower_bound,
            'short_upper_bound': short_upper_bound,
            'min_word': min_word,
            'min_prob': min_prob,
            'log_scale': log_scale}

    return cfg

def get_rescore_cfg(decode_dir,
                    lm_csv_name,
                    hyp_search,
                    equation = 6,
                    text_norm = 'custom',
                    preds_score_csv_name = 'preds_score.csv')
    preds_score_csv = os.path.join(decode_dir,preds_score_csv_name)
    lm_csv = os.path.join(decode_dir, lm_csv_name)
    cfg = {'decode_dir':decode_dir,
            'equation': equation, 
            'text_norm': text_norm,
            'preds_score_csv': preds_score_csv,
            'lm_csv': lm_csv,
            'hyp_search': hyp_search
          }
    return cfg

def get_cfg1(asr_params,
            lm_params,
            dataset_params,
            lm_score_params,
            rescore_params,
            hyper_params):
    if asr_params:
        cfg_decode_asr = get_decode_cfg_asr(**asr_params)
    else:
        cfg_decode_asr = get_decode_cfg_asr()

    cfg_lm = get_lm_cfg(**lm_params)
    cfg_decode_dataset = get_decode_cfg_dataset(**dataset_params)
    lm_score_cfg_dataset = get_lm_score_cfg_dataset(**lm_score_params)
    hyper_params_cfg = get_hyper_params_cfg(**hyper_params)
    rescore_cfg = get_rescore_cfg(**rescore_params)
    
    decode_cfg = {**cfg_decode_asr, **cfg_decode_dataset}
    lm_score_cfg = {**cfg_lm, **lm_score_cfg_dataset}
    if rescore_params['hyp_search']:
        rescore_cfg.update(hyper_params_cfg)
    return decode_cfg, lm_score_cfg, rescore_cfg
