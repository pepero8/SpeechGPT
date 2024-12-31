import sys
sys.path.append("/home/jhwan98/EmoSDS/SpeechGPT") # added by jaehwan
import torch
import torch.nn as nn
# from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import soundfile as sf
from typing import List
import argparse
import logging
import json
from tqdm import tqdm
import os
import re
import traceback
from peft import PeftModel
from speechgpt.utils.speech2unit.speech2unit import Speech2UnitCustom
import transformers
from transformers import (
    AutoConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer, GenerationConfig,
)


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NAME="EmoSDS"
# META_INSTRUCTION = ""  # step 1 style prediction
# META_INSTRUCTION = "You are a human-like dialogue agent EmoSDS that imitates real human-to-human spoken dialogue. The speaking style should be very natural in the dialogue context. Generate human-like response given human speech. " # step 2 dialogue prediction
META_INSTRUCTION = f"""
# Task
From now on, you are an intelligent voice assistant. You need to provide useful, consistent to the dialogue context, emotionally approval natural response to the user's input speech.
Given user speech, you need to transcribe the user speech, identify the speaking style, predict appropriate response style, and predict appropriate response text according to the response style.
The speaking style should be one of following 11 styles: neutral, angry, cheerful, sad, excited, friendly, terrified, shouting, unfriendly, whispering, hopeful

# Examples
Following examples show example responses to the transcribed input speech with speaking style. The caption in square brackets indicate speaking style of the transcription."

## Example 1
Input: [excited] I can't believe it's not butter!
Answer: [friendly] Oh wow, you're really passionate about this! So, what is it about "I Can't Believe It's Not Butter" that's got you so excited?

## Example 2
Input: [angry] I can't believe it's not butter!
Answer: [neutral] Whoa, okay, let's take a deep breath and try to calm down. Are you actually upset that it's not butter? What's really going on here?

## Example 3
Input: [neutral] I watched a baseball game on the weekend
Answer: [friendly] Oh cool! how was it?

## Example 4
Input: [sad] I watched a baseball game on the weekend
Answer: [neutral] You don't seem too happy, did your team lose?"""  # step 2 dialogue prediction

# USER_INSTRUCTION = "Identify speaking style of given speech: {units}. Provide only the style label > ["
DEFAULT_GEN_PARAMS = { # following Spoken-LLM
        "max_new_tokens": 1024,
        "min_new_tokens": 10,
        "temperature": 0.7,
        "do_sample": True, 
        "top_k": 60,
        "top_p": 0.95,
        }  
device = torch.device('cuda')


def extract_text_between_tags(text, tag1='[EmoSDS] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response


class SpeechGPTInference:
    def __init__(
        self, 
        model_name_or_path: str,
        lora_weights: str=None,
        s2u_dir: str="speechgpt/utils/speech2unit/",
        vocoder_dir: str="speechgpt/utils/vocoder/", 
        output_dir="speechgpt/output/"
        ):

        self.meta_instruction = META_INSTRUCTION
        # self.template= "[Human]: {question} <eoh>. [SpeechGPT]: "
        # self.template = "Identify speaking style of given speech: {units}. Provide only the style label > ["  # step 1 style prediction
        # self.template = "[Human]:Identify speaking style of given speech: {units}. Provide only the style label without any explanation<eoh>.[EmoSDS]:["  # step 1 style prediction
        # self.template = "[Human]:{units}<eoh>.["  # step 2 dialogue prediction
        self.template = "Input: {units} ["  # step 2 dialogue prediction

        # speech2unit
        self.s2u = Speech2UnitCustom(ckpt_dir=s2u_dir)

        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if lora_weights is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.model.half()  

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 

        # generation
        self.generate_kwargs = DEFAULT_GEN_PARAMS

        # vocoder
        # vocoder = os.path.join(vocoder_dir, "vocoder.pt")
        # vocoder_cfg = os.path.join(vocoder_dir, "config.json")
        # with open(vocoder_cfg) as f:
        #     vocoder_cfg = json.load(f)
        # self.vocoder = CodeHiFiGANVocoder(vocoder, vocoder_cfg).to(device)

        self.output_dir = output_dir

    def preprocess(
        self,
        raw_text: str,
    ):
        processed_parts = []
        for part in raw_text.split("is input:"):
            # for _ in raw_text:
            # if os.path.isfile(part.strip()) and os.path.splitext(part.strip())[-1] in [".wav", ".flac", ".mp4"]:
            if os.path.isfile(part.strip()) and os.path.splitext(part.strip())[-1] in [
                ".wav",
                ".flac",
                ".mp4",
            ]:
                processed_parts.append(self.s2u(part.strip(), merged=True))
            else:
                # processed_parts.append(part)
                continue
                # raise Exception(f"audio file not found: {part.strip()}")
        # processed_text = "is input:".join(processed_parts)
        processed_text = "".join(processed_parts)

        prompt_seq = self.meta_instruction + self.template.format(units=processed_text)
        return prompt_seq

    def postprocess(
        self,
        response: str,
    ):

        question = extract_text_between_tags(response, tag1="[Human]", tag2="<eoh>")
        answer = extract_text_between_tags(response + '<eoa>', tag1=f"[SpeechGPT] :", tag2="<eoa>")
        tq = extract_text_between_tags(response, tag1="[SpeechGPT] :", tag2="; [ta]") if "[ta]" in response else ''
        ta = extract_text_between_tags(response, tag1="[ta]", tag2="; [ua]") if "[ta]" in response else ''
        ua = extract_text_between_tags(response + '<eoa>', tag1="[ua]", tag2="<eoa>") if "[ua]" in response else ''

        return {"question":question, "answer":answer, "textQuestion":tq, "textAnswer":ta, "unitAnswer":ua}

    def forward(
        self, 
        prompts: List[str]
    ):
        with torch.no_grad():
            # preprocess
            preprocessed_prompts = []
            for prompt in prompts:
                preprocessed_prompts.append(self.preprocess(prompt))

            input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
            for input_id in input_ids:
                if input_id[-1] == 2:
                    input_id = input_id[:, :-1]

            input_ids = input_ids.to(device)

            # generate
            # generation_config = GenerationConfig(
            #     temperature=0.7,
            #     top_p=0.8,
            #     top_k=50,
            #     do_sample=True,
            #     max_new_tokens=2048,
            #     min_new_tokens=10,
            #     )
            generation_config = GenerationConfig(
                temperature=self.generate_kwargs["temperature"],
                top_p=self.generate_kwargs["top_p"],
                top_k=self.generate_kwargs["top_k"],
                do_sample=self.generate_kwargs["do_sample"],
                max_new_tokens=self.generate_kwargs["max_new_tokens"],
                min_new_tokens=self.generate_kwargs["min_new_tokens"],
            )

            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                # max_new_tokens=1024,
            )
            generated_ids = generated_ids.sequences
            responses = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)

            print("[", responses, "]")

            # # postprocess
            # responses = [self.postprocess(x) for x in responses]

            # # save repsonses
            # init_num = sum(1 for line in open(f"{self.output_dir}/responses.json", 'r')) if os.path.exists(f"{self.output_dir}/responses.json") else 0
            # with open(f"{self.output_dir}/responses.json", 'a') as f:
            #     for r in responses:
            #         if r["textAnswer"] != "":
            #             print("Transcript:", r["textQuestion"])
            #             print("Text response:", r["textAnswer"])
            #         else:
            #             print("Response:\n", r["answer"])
            #         json_line = json.dumps(r)
            #         f.write(json_line+'\n')

            # # dump wav
            # # wav = torch.tensor(0)
            # # os.makedirs(f"{self.output_dir}/wav/", exist_ok=True)
            # # for i, response in enumerate(responses):
            # #     if response["answer"] != '' and '<sosp>' in response["answer"]:
            # #         unit = [int(num) for num in re.findall(r'<(\d+)>', response["answer"])]
            # #         x = {
            # #                 "code": torch.LongTensor(unit).view(1, -1).to(device),
            # #             }
            # #         wav = self.vocoder(x, True)
            # #         self.dump_wav(init_num+i, wav, prefix="answer")
            # #         print(f"Speech repsonse is saved in {self.output_dir}/wav/answer_{init_num+i}.wav")
            # print(f"Response json is saved in {self.output_dir}/responses.json")

        # return 16000, wav.detach().cpu().numpy()
        return

    def dump_wav(self, sample_id, pred_wav, prefix):
        sf.write(
            f"{self.output_dir}/wav/{prefix}_{sample_id}.wav",
            pred_wav.detach().cpu().numpy(),
            16000,
        )

    def __call__(self, input):
        return self.forward(input)

    def interact(self):
        prompt = str(input(f"Please talk with {NAME}:\n"))
        while prompt != "quit":
            try:
                self.forward([prompt])
            except Exception as e:
                traceback.print_exc()
                print(e)

            prompt = str(input(f"Please input prompts for {NAME}:\n"))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="")
    parser.add_argument("--lora-weights", type=str, default=None)
    parser.add_argument("--s2u-dir", type=str, default="speechgpt/utils/speech2unit/")
    parser.add_argument("--vocoder-dir", type=str, default="speechgpt/utils/vocoder/")
    parser.add_argument("--output-dir", type=str, default="speechgpt/output/")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)


    infer = SpeechGPTInference(
        args.model_name_or_path,
        args.lora_weights,
        args.s2u_dir,
        args.vocoder_dir,
        args.output_dir
    )

    infer.interact()
