import json
import logging
import re
from pathlib import Path
from typing import Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_input_json(file_path: str) -> Dict[str, Any]:
    """Load the validation.json data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()  # model info
        return f.read().strip()


def output_cqs(text: str, prompt: str, model, tokenizer) -> str:
    instruction = prompt.format(intervention=text)
    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split(instruction)[-1].strip()


def structure_output(raw_text: str) -> list:
    logger.debug("=== Raw model output ===")
    logger.debug(raw_text)
    logger.debug("=========================")

    cqs_list = raw_text.strip().split('\n')
    final = []
    valid = []
    not_valid = []

    for cq in cqs_list:
        if re.match(r'.*\?(")? ?(\([a-zA-Z0-9\.\'\-,\? ]*\))?([a-zA-Z \.,"\']*)?"?$', cq):
            valid.append(cq)
        else:
            not_valid.append(cq)

    for text in not_valid:
        new_cqs = re.split(r'\?"', text + 'end')
        if len(new_cqs) > 1:
            valid.extend([cq + '?"' for cq in new_cqs[:-1]])

    for cq in valid:
        occurrence = re.search(r'[A-Z]', cq)
        if occurrence:
            final.append(cq[occurrence.start():])

    if len(final) >= 3:
        return [{"id": i, "cq": final[i]} for i in range(3)]
    else:
        logger.warning("Missing CQs")
        return "Missing CQs"


def main():
    # Load prompt and model name from prompt text file
    with open('fewshot_prompt.txt', 'r', encoding='utf-8') as file: # Changing the prompt file name to either zeroshot or few shot
        first_line = file.readline()
        model_name = first_line.split('=')[1].strip()
        prompt = file.read().strip()

    # Load validation input data
    with open('data/validation.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Using model: {model_name}")
    generation_config = GenerationConfig.from_pretrained(model_name)
    logger.info(generation_config)

    # Load model and tokenizer
    if 'llama' in model_name.lower():
        model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Loaded {model_name}")

    output_data = {} # Create an empty dictionary to store the output

    # Generate CQs for each intervention
    for item in tqdm(data.values(), desc="Generating CQs"):
        intervention_id = item["intervention_id"]
        text = item["intervention"]

        try:
            cqs_raw = output_cqs(text, prompt, model, tokenizer)
            logger.info("Raw model output:\n%s", cqs_raw)
            item["cqs"] = structure_output(cqs_raw)
        except Exception as e:
            logger.error(f"Failed to generate CQs for {intervention_id}: {e}")
            item["cqs"] = "Missing CQs"

        output_data[intervention_id] = item

    # Save the output to a JSON file
    # --- Changing this to work for Windows--- #
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    output_filename = f'output_fewShot_{safe_model_name}.json'

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    logger.info(f'Saved output to {os.path.abspath(output_filename)}')



if __name__ == "__main__":
    main()
