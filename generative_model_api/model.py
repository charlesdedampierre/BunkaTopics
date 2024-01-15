import torch
from langchain import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


# function for loading 4-bit quantized model
def load_quantized_model(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=False,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    return model


# fucntion for initializing tokenizer
def initialize_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer


if __name__ == "__main__":
    # Chose and load model
    model_name = "anakin87/zephyr-7b-alpha-sharded"
    tokenizer = initialize_tokenizer(model_name)
    model = load_quantized_model(model_name)

    pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2048,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipeline)

    prompt_template = "What is a good name for a company that makes {product}?"

    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    llm_chain("colorful socks")
