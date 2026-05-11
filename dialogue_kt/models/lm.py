import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from dialogue_kt.utils import get_checkpoint_path

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def get_base_model(base_model_name: str, tokenizer: AutoTokenizer, quantize: bool):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=bnb_config if quantize else None,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    return base_model


def get_model(base_model_name: str, test: bool,
              model_name: str = None, pt_model_name: str = None,
              r: int = None, lora_alpha: int = None,
              quantize: bool = True, use_gradient_checkpointing: bool = True):

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.bos_token

    model = get_base_model(base_model_name, tokenizer, quantize)

    if test and model_name:
        print(f"Loading fine-tuned adapter: {model_name}")
        model = PeftModel.from_pretrained(model, get_checkpoint_path(model_name))
    elif not test:
        if quantize:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=use_gradient_checkpointing
            )
        if pt_model_name:
            print(f"Loading pre-trained adapter: {pt_model_name}")
            model = PeftModel.from_pretrained(
                model, get_checkpoint_path(pt_model_name),
                is_trainable=True, adapter_name="default"
            )
        else:
            print(f"Initializing new LoRA adapters (r={r}, alpha={lora_alpha})")
            peft_config = LoraConfig(
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                r=r if r else 8,
                lora_alpha=lora_alpha if lora_alpha else 16,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
            model = get_peft_model(model, peft_config)
    else:
        print("Zero-shot inference mode")

    return model, tokenizer