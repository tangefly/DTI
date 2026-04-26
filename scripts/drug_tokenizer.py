from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../tokenizers/drug")

token_list = tokenizer.tokenize("[H][C@@]1(OC(=O)C(O)=C1O)[C@@H](O)CO")
id_list = tokenizer.convert_tokens_to_ids(token_list)

print(token_list)
print(id_list)
