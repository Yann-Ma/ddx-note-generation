import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from utils.generate_glm import generate as generate_off
from utils.generate_glm import load_data as load_data_private
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    data_private = load_data_private()

    tokenizer = AutoTokenizer.from_pretrained(r"./models/glm-4-9b/", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(r"./models/glm-4-9b/", trust_remote_code=True, torch_dtype=torch.bfloat16)

    generate_off(tokenizer, model, data_private, "basic")
    generate_off(tokenizer, model, data_private, "with_final_diag")
    generate_off(tokenizer, model, data_private, "with_format_ctrl")
    generate_off(tokenizer, model, data_private, "with_fd_fc")

main()