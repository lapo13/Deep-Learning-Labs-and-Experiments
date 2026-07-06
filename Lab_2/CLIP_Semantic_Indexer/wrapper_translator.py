from transformers import AutoModelForCausalLM, AutoTokenizer
import os, torch

class wrapper_translator:
    def __init__(self, model_name: str, model_save_path: str = "./translation_model"):
        if not os.path.exists(model_save_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
            self.model.save_pretrained(model_save_path)
            self.tokenizer.save_pretrained(model_save_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_save_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_save_path)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device) # type: ignore

    def translate(self, testo_italiano):
        with torch.no_grad():
            messages = [
                {"role": "system", "content": "You are an expert AI prompt engineer for image retrieval models (CLIP)."
                "Your task is to translate the user's input from Italian to English AND clean it."
                "Extract ONLY the core visual objects, characters, actions, or scenes mentioned. "
                "Stritly remove all conversational filler, metadata, polite phrases, or ambiguous words.Output ONLY the final translated and cleaned English description. "
                "Do not include introductions, explanations, quotes, or punctuation like periods at the end. Be concise and direct."},
                {"role": "user", "content": testo_italiano}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(**model_inputs, max_new_tokens=50, temperature=0.1) # Temperatura bassa = zero fantasia
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()