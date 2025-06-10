from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
import torch

class LLMLoader:

    def __init__(self, model_name: str, token: str = ""):
        self.model_name = model_name
        self.token = token

    def load_quantized_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.token)
        tokenizer.pad_token = tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.token,
            quantization_config=quant_config,
            device_map="balanced"
        )

        return model, tokenizer

    def load_qlora_model(self):

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.token)
        tokenizer.pad_token = tokenizer.eos_token

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.token,
                quantization_config=quant_config,
                device_map="balanced"
        )

        if "Falcon3-Mamba" in self.model_name:
            target_modules = ['in_proj', 'dt_proj']
        else:
            target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none"
        )

        qlora_model = get_peft_model(model, lora_config)

        return qlora_model, tokenizer


class Loader:
    model_loader = {
        "simple": "load_simple_model",
        "pipeline": "load_pipeline_model",
        "qlora": "load_qlora_model",
        "quantized": "load_quantized_model"
    }

    def __new__(cls, model_name: str, token: str = "", load_type: str = "qlora"):
        loader = "LLMLoader(model_name=model_name, token=token)." + Loader.model_loader.get(load_type,
                                                                                            "load_simple_model") + "()"
        return eval(loader)



class JediQA:

    def __init__(self, model_name, load_type="qlora", token=""):
        self.model, self.tokenizer = Loader(model_name=model_name, load_type=load_type, token=token)

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer

    def answer(self, input_text, max_new_tokens=512):
        encoded_data = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **encoded_data,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(encoded_data["input_ids"], generated_ids)
        ]
        decoded_data = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return decoded_data

    def answer_batch(self, input_texts, max_new_tokens=512):
        encoded_data = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **encoded_data,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(encoded_data["input_ids"], generated_ids)
        ]
        decoded_data = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded_data