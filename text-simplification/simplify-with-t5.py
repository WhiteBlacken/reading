
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

if __name__ == '__main__':
    input_text = "这是一个较长的文本，需要被简化成一个简短的文本。"
    input_ids = tokenizer.encode(f"summarize: {input_text}", return_tensors='pt')
    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)



