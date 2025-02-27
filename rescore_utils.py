import torch
import numpy as np
import os
from utils import wer_wrap, save_df

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

def rescore_lm(preds_beam,preds,model, tokenizer):
    preds_rescore = []
    prob_sequences = []
    for i in range(len(preds_beam)):
        candidates =  [preds[i]] + preds_beam[i]
        prob_sequence = [logprobs(model, tokenizer, candidate)[0] for candidate in candidates if str(candidate).strip()]
        best_candidate = candidates[int(np.argmax(prob_sequence))]
        preds_rescore.append(best_candidate)
        prob_sequences.append(prob_sequence)
    return preds_rescore, prob_sequences

def cps_penalty(x, center = 12):
    # Penalty when number of character per second is far away from the mean. Parabol shape
    # Assumption: average character per second is 12
    return -(x - center) ** 2

def wps_penalty(x, center = 3.5):
    # Penalty when number of words per second is far away from the mean. Parabol shape
    # Assumption: average character per second is 3.5
    return -(x - center) ** 2

def count_chars(text):
    # count number of characters without space
    return len(''.join(text.rstrip().split()))

def count_word(text):
    return len(text.split(' '))

def rescore_equation_6(preds,
                       preds_score,
                       lm_scores,
                       preds_beam, 
                       preds_beam_score,
                       lm_beam_scores,
                       wavs_len, 
                       asr_w,
                       lm_w,
                       short_w,
                       min_word=0,
                       min_prob=1):
    preds_rescore = []
    for i in range(len(preds)):
        candidates =   [preds[i]] + preds_beam[i]
        num_words = np.array([count_word(cand) for cand in candidates])
        # If the greedy candidate/hypothesis is not a strong candidate, then search for the candidate among the beam candidates and the greedy candidate
        if num_words[0]/wavs_len[i] < min_word or np.exp(preds_score[i]) < min_prob:
            logprob_seq_asr =  np.array([preds_score[i]] + preds_beam_score[i])
            logprob_seq_lm  =  np.array([lm_scores[i]]   + lm_beam_scores[i])
            short_pen= np.array([wps_penalty(num/(wavs_len[i])) for num in num_words])
            logprob_seq = asr_w * logprob_seq_asr + lm_w * logprob_seq_lm  + short_w * short_pen
            best_candidate = candidates[int(np.argmax(logprob_seq))]
            preds_rescore.append(best_candidate)
        # if the greedy is a strong candidate, just take it
        else:
            preds_rescore.append(preds[i])
    return preds_rescore

def evaluate_wer(preds,
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
                **kwargs):
    if equation == 6:
        preds_rescore = rescore_equation_6( preds,
                                            preds_score,
                                            lm_scores,
                                            preds_beam, 
                                            preds_beam_score,
                                            lm_beam_scores,
                                            wavs_len, 
                                            kwargs['asr_weight'],
                                            kwargs['lm_weight'],
                                            kwargs['short_penalty_weight'],
                                            kwargs['min_word'],
                                            kwargs['min_prob'])  
 
    os.makedirs(rescore_dir, exist_ok=True)
    data_dict = {
        'wavs_file': wavs_file,
        'preds_rescore': preds_rescore
    }
    save_df(os.path.join(rescore_dir, 'preds_rescore.csv'), data_dict)
    labels_norm_filter, preds_norm_rescore_filter, preds_beam_norm_filter, wavs_file_filter, wer_rescore, cer_rescore = wer_wrap(labels,
                                                                                                                                 preds_rescore,
                                                                                                                                 preds_beam,
                                                                                                                                 normalizer,
                                                                                                                                 os.path.join(rescore_dir,'wer_rescore.txt'),
                                                                                                                                 os.path.join(rescore_dir,'cer_rescore.txt'),
                                                                                                                                 wavs_file)
    return wer_rescore, labels_norm_filter, preds_norm_rescore_filter, preds_beam_norm_filter, wavs_file_filter, wer_rescore, cer_rescore

def get_hyp_search_params(**config):
    if config['equation'] == 6 :
        params = [
                    {
                        "name": "asr_weight",  # The name of the parameter.
                        "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                        "bounds": [config['asr_lower_bound'],config['asr_upper_bound']],  # The bounds for range parameters. 
                        "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                        "log_scale": config['log_scale'] 
                    },
                    {
                        "name": "lm_weight",  # The name of the parameter.
                        "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                        "bounds": [config['lm_lower_bound'],config['lm_upper_bound']],  # The bounds for range parameters. 
                        "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                        "log_scale": config['log_scale'] 
                    },
                    {
                        "name": "short_penalty_weight",  # The name of the parameter.
                        "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                        "bounds": [config['short_lower_bound'],config['short_upper_bound']],  # The bounds for range parameters. 
                        "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                        "log_scale": config['log_scale'] 
                    },
                    {
                        "name": "min_word",  # The name of the parameter.
                        "type": "fixed",  # The type of the parameter ("range", "choice" or "fixed"). 
                        "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                        "value" : config["min_word"], 
                    },
                    {
                        "name": "min_prob",  # The name of the parameter.
                        "type": "fixed",  # The type of the parameter ("range", "choice" or "fixed"). 
                        "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                        "value" : config["min_prob"], 
                    }
                ]
    return params

def get_rescore_params(**kwargs):
    if kwargs['equation'] == 6:
        params = { 'asr_weight': kwargs['asr_weight'],
                  'lm_weight' : kwargs['lm_weight'],
                  'short_penalty_weight': kwargs['short_penalty_weight'],
                  'min_word' : kwargs['min_word'],
                  'min_prob'  : kwargs['min_prob']
                   }
    return params
