from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../tokenizers/protein")

token_list = tokenizer.tokenize("ACDEFGHIK")
id_list = tokenizer.convert_tokens_to_ids(token_list)

print(token_list)
print(id_list)

