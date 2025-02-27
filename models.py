import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
from peft import PeftModel, PeftConfig
import os

def load_whisper_finetune(asr_model_name_or_path, device='cuda', language='english',task='transcribe'):
    checkpoint_dir = os.path.expanduser(asr_model_name_or_path)
    lang='english'
    init_from_hub_path = f"openai/whisper-large-v2"
    USE_INT8 = False
    if os.path.isdir(os.path.join(checkpoint_dir , "adapter_model")):
        print('...it looks like this model was tuned using PEFT, because adapter_model/ is present in ckpt dir')

        # checkpoint dir needs adapter model subdir with adapter_model.bin and adapter_confg.json
        peft_config = PeftConfig.from_pretrained(os.path.join(checkpoint_dir , "adapter_model"))
        # except ValueError as e: # if final checkpoint these are in the parent checkpoint direcory
        #     peft_config = PeftConfig.from_pretrained(os.path.join(checkpoint_dir ), subfolder=None)
        model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, 
            load_in_8bit=USE_INT8, 
            device_map='auto', 
            use_cache=False,
            )
        model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir,"adapter_model"))
    tokenizer = WhisperTokenizer.from_pretrained(init_from_hub_path, language=lang, task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(init_from_hub_path)
    processor = WhisperProcessor.from_pretrained(init_from_hub_path, language=lang, task="transcribe")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    return model, tokenizer, processor, feature_extractor, forced_decoder_ids
def load_whisper(asr_model_name_or_path='openai/whisper-large-v2', device='cuda', language='english',task='transcribe'):
    model = WhisperForConditionalGeneration.from_pretrained(asr_model_name_or_path)
    model = model.to(device)
    model.eval()
    tokenizer = WhisperTokenizer.from_pretrained(asr_model_name_or_path, language=language)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(asr_model_name_or_path)
    processor = WhisperProcessor.from_pretrained(asr_model_name_or_path)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    return model, tokenizer, processor, feature_extractor, forced_decoder_ids

def load_gpt2(model_name_or_path,device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer

def load_llama(model_name_or_path,device='cuda'):
    # e.g model_name_or_path = '/home/vtrinh/download/llama2_huggingface_converted/7B'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    model.eval()
    return model, tokenizer

def load_llama2_peft(peft_model_name_or_path, device='cuda'):
    peft_config = PeftConfig.from_pretrained(peft_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path,
                                                load_in_8bit=True, 
                                                device_map="auto")
    model = PeftModel.from_pretrained(model, peft_model_name_or_path)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path,padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_lm(lm, model_name_or_path, device ='cuda'):
    if lm in['gpt2','gemma','mistral']:
        return load_gpt2(model_name_or_path,device=device)
    elif lm == 'llama2_peft':
        return load_llama2_peft(model_name_or_path, device=device)
    
def load_tokenizer(lm, model_name_or_path):
    if lm == 'llama2_peft':
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token
    elif lm in['gpt2','gemma','mistral']:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

