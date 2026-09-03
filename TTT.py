import tiktoken

full_prompt = "Your full prompt text goes here."

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
content_tokens = len(encoding.encode(full_prompt))

print(encoding.encode(full_prompt))
print(f"Number of tokens in the full prompt: {content_tokens}")