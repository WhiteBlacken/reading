from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("husseinMoh/bart-base-finetuned-text-simplification")

model = AutoModelForSeq2SeqLM.from_pretrained("husseinMoh/bart-base-finetuned-text-simplification")

if __name__ == '__main__':
    ARTICLE_TO_SUMMARIZE = "While pondering over the intricacies of linguistics and the convolutions of grammar, the linguist, having been deep in thought for hours, inadvertently overlooked the time and missed his appointment with the head of the department, much to his chagrin and embarrassment. "
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")
    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=0, max_length=20)
    print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])