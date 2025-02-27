# Path
export PYTHONPATH="/home/vtrinh/projects:$PYTHONPATH"
# To run for GPT-2, first update the config then
python runs.py --config /home/vtrinh/projects/rescore_asr/configs/exp/levi/gpt2/gpt2_exp_01.yaml
# Similarly to run for Llama-2:
python runs.py --config /home/vtrinh/projects/rescore_asr/configs/exp/levi/llama2/exp_01_llama2.yaml