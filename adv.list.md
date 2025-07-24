1. zero locating using multi centered voxels
        https://github.com/ConsistentlyInconsistentYT/Pixeltovoxelprojector


2. Unsloth Multi-gpu
        https://github.com/anhvth/opensloth
        https://github.com/thad0ctor/unsloth-5090-multiple
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    load_in_4bit = True,
    device_map = "balanced",
)
```
