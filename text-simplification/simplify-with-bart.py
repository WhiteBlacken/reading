from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("twigs/bart-text2text-simplifier")

model = AutoModelForSeq2SeqLM.from_pretrained("twigs/bart-text2text-simplifier")

if __name__ == '__main__':
    ARTICLE_TO_SUMMARIZE = "Safe House,starring Denzel Washington and Ryan Reynolds,is a 2012 South African & American action thriller film directed by Daniel Espinosa"
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")
    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=0, max_length=20)
    print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
