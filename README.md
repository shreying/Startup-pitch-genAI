# Startup Pitch Generation Using LoRA Fine-Tuned GPT-2

This repository provides a fine-tuned version of GPT-2 using Low-Rank Adaptation (LoRA) for generating startup pitches. The model is trained on a custom dataset of startup pitches, which is available in the repository. The goal of this project is to generate high-quality and relevant startup pitches based on user input, enhancing the startup idea generation process.

---

## Table of Contents

- [Overview](#overview)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The project leverages the GPT-2 model fine-tuned using LoRA for startup pitch generation. LoRA helps in fine-tuning large pre-trained models like GPT-2 with a lower computational cost. The fine-tuned model can generate creative and coherent startup pitches based on user input, making it useful for entrepreneurs and business developers.

![Streamlit Interface](https://github.com/shreying/Startup-pitch-genAI/blob/main/streamlit%20interface.png)


---

## Model Details

### Model Description

The GPT-2 model has been fine-tuned with Low-Rank Adaptation (LoRA) to generate startup pitches. The model has been trained on a custom dataset that includes various startup ideas and their corresponding pitches. It is based on the GPT-2 architecture, which is known for its ability to generate coherent and creative text.

- **Developed by:** Shreya Sahu
- **Model type:** GPT-2 with LoRA fine-tuning
- **Language(s):** English
- **License:** MIT License
- **Fine-tuned from model:** GPT-2

---

## Installation

### Prerequisites

Ensure that you have Python 3.7+ and the required dependencies installed. You can install the dependencies by running the following command:

```bash
pip install -r requirements.txt
````

### Installing Dependencies

 You can manually install the necessary libraries like:

```bash
pip install transformers datasets torch peft
```
or refer to the jupyter notebook [here](https://github.com/shreying/Startup-pitch-genAI/blob/main/ProjectAI.ipynb).

---

## Usage

To generate a startup pitch, run the following command:

```bash
streamlit run pitchgenerator.py
```

This will generate a startup pitch based on the input provided.

---

## Dataset

The model has been fine-tuned on a custom dataset of startup pitches, which can be found in the file [startup\_pitches.jsonl](https://github.com/shreying/Startup-pitch-genAI/blob/main/startup_pitches.jsonl).

### Dataset Overview

* **File format:** JSON Lines (.jsonl)
* **Contents:** The dataset contains multiple startup ideas and their corresponding startup pitches.

You can view the dataset file [here](https://github.com/shreying/Startup-pitch-genAI/blob/main/startup_pitches.jsonl).

---

## Training

The model has been fine-tuned using the following hyperparameters:

* **Model architecture:** GPT-2
* **Fine-tuning method:** Low-Rank Adaptation (LoRA)
* **Training dataset:** `startup_pitches.jsonl`
* **Training procedure:** The model was trained on a standard language modeling task using the dataset of startup pitches.

The training code is available in the [training notebook](https://github.com/shreying/Startup-pitch-genAI/blob/main/ProjectAI.ipynb). You can open this Jupyter Notebook to see the details of the training process, including how the dataset is loaded, the LoRA fine-tuning process, and the hyperparameters used.

---

## Evaluation

The model was evaluated using a set of metrics that assess its ability to generate relevant, coherent, and creative startup pitches. Evaluation results will be available soon.

---

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please feel free to create an issue or a pull request.

To contribute:

1. Fork the repository.
2. Clone your forked repository.
3. Create a new branch for your changes.
4. Make your changes and commit them.
5. Push your changes to your fork.
6. Open a pull request to the main repository.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

* GPT-2 and LoRA for providing the foundation for this model.
* [Hugging Face](https://huggingface.co/) for their excellent library and tools that made it easier to fine-tune and use the GPT-2 model.

---

## Citation

If you use this model or dataset in your work, please cite it as follows:

```bibtex
@misc{startup-pitch-genai,
  author = {Shreya Sahu},
  title = {LoRA Fine-Tuned GPT-2 for Startup Pitch Generation},
  year = {2025},
  howpublished = {\url{https://github.com/shreying/Startup-pitch-genAI}}
}
```

---

## Contact

For any questions or inquiries, feel free to reach out to [Shreya Sahu](mailto:sahush2004@gmail.com).

