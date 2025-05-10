---
base_model: gpt2
library_name: peft
---
# Model Card for Startup Pitch Generator (LoRA fine-tuned GPT-2)

This model is a fine-tuned version of `gpt2` using the LoRA method via the PEFT library. It is designed to generate startup pitch content given a prompt related to a business idea, product, or sector.

---

## Model Details

### Model Description

This GPT-2 based model has been fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA (Low-Rank Adaptation), on a dataset of curated startup pitches and business descriptions. It can assist entrepreneurs, students, and creators in generating structured pitch content for presentations, hackathons, or incubation programs.

- **Developed by:** Shreya Sahu  
- **Model type:** LoRA fine-tuned language model  
- **Language(s):** English  
- **License:** MIT  
- **Finetuned from model:** [`gpt2`](https://huggingface.co/gpt2)

### Model Sources

- **Repository:** [https://github.com/shreying/Startup-pitch-genAI](https://github.com/shreying/Startup-pitch-genAI)

---

## Uses

### Direct Use

- Generate startup pitch ideas from short prompts
- Expand brief product descriptions into structured presentations
- Support brainstorming and content creation for entrepreneurial use cases

### Downstream Use

- Integration into web tools for startup pitch assistance
- Fine-tuning further for domain-specific applications (e.g., healthtech, edtech)

### Out-of-Scope Use

- Not suitable for generating legally binding documents
- Not intended for generating pitches without review or human supervision

---

## Bias, Risks, and Limitations

This model may reproduce biases present in startup-related training data (e.g., region or gender biases). It may sometimes hallucinate facts or make unrealistic business claims.

### Recommendations

Review and edit generated content before use. Use it as a writing assistant rather than a sole content creator.

---

## How to Get Started with the Model

You can run the model using the following PEFT and Transformers setup:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("path_to_lora_adapter")
base_model = GPT2LMHeadModel.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, "path_to_lora_adapter")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_text = "A wearable device that tracks hydration levels"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


## Training Details

### Training Data

JSONL file containing the pitches to specific prompts.
You can find the dataset file [here](https://github.com/shreying/Startup-pitch-genAI/blob/main/startup_pitches.jsonl).



### Training Procedure

Preprocessing
- Tokenized using GPT-2 tokenizer
- Cleaned for formatting consistency

Training Hyperparameters
- Precision: fp16 mixed precision
- Epochs: 3
- Batch size: 8
- Learning rate: 5e-5


## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

Held-out subset of pitch prompts unseen during training.

#### Factors

- Prompt diversity (industry, length)
- Coherence, relevance, business clarity

#### Metrics

- Manual evaluation (coherence, novelty)
- Perplexity (optional)


### Results

- Generates creative and structured pitch formats
- Better than base GPT-2 on pitch-specific prompts



#### Summary


## Environmental Impact

- Hardware Type: NVIDIA T4 GPU (Google Colab)
- Hours used: ~2 hours
- Cloud Provider: Google
- Compute Region: Asia-South1
- Carbon Emitted: Minimal (low training duration and hardware)

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Single GPU (T4)
- **Software:** Transformers, PEFT, Accelerate, PyTorch


## Citation [optional]


**BibTeX:**

@misc{startup-pitch-genai,
  author = {Shreya Sahu},
  title = {LoRA Fine-Tuned GPT-2 for Startup Pitch Generation},
  year = {2025},
  howpublished = {\url{https://github.com/shreying/Startup-pitch-genAI}}
}


## Model Card Authors 

Author: Shreya Sahu
Email: sahush2004@gmail.com
GitHub: shreying


### Framework versions

- PEFT 0.15.2
- Transformers 4.39+
- PyTorch 2.x
