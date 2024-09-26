import os

from huggingface_hub import login
from transformers import (AutoModelForCausalLM, AutoTokenizer)

from cenai_core import (cenai_path, load_dotenv)


class Quantizer:
    def __init__(
        self,
        model_id: str,
        body_name: str
    ):
        load_dotenv()
        login(token=os.environ["HF_TOKEN"])

        self._model_id = model_id
        self._body_name = body_name

    def quantize(self) -> None:
        for bit_num in [8, 4]:

            model_path = cenai_path(
                f"model/{self._body_name}-Q{bit_num}"
            )

            print(f"{self._model_id} quantized to Q{bit_num} ....")

            tokenizer = AutoTokenizer.from_pretrained(self._model_id)

            llm = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                load_in_8bit=(bit_num == 8),
                load_in_4bit=(bit_num == 4),
                device_map="auto",
            )

            print(f"{self._model_id} quantized to Q{bit_num} DONE")
        
            print(f"quantized model saved to {model_path} ....")

            llm.save_pretrained(str(model_path))
            tokenizer.save_pretrained(str(model_path))

            print(f"quantized model saved to {model_path} DONE")

    def push(self, user: str) -> None:
        for bit_num in [8, 4]:

            model_path = cenai_path(
                f"model/{self._body_name}-Q{bit_num}"
            )

            if not model_path.is_dir():
                continue

            print(f"quantized model loaded from {model_path} ....")

            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            llm = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
            )

            print(f"quantized model loaded from {model_path} DONE")

            print(f"quantized model pushed to {user}/{model_path.name} ....")

            llm.push_to_hub(f"{user}/{model_path.name}")
            tokenizer.push_to_hub(f"{user}/{model_path.name}")

            print(f"quantized model pushed to {user}/{model_path.name} DONE")


if __name__ == "__main__":
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    body_name = "meta-llama-3.1-8B"

    quantizer = Quantizer(
        model_id=model_id,
        body_name=body_name,
    )

#   quantizer.quantize()
    quantizer.push("hmyang71")
