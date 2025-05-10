import streamlit as st
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch

# Avoid torch class introspection issues
torch.classes = None

# Set a consistent seed for reproducibility
from transformers import set_seed
set_seed(42)

# Load your fine-tuned GPT-2 model and tokenizer
@st.cache_resource
def load_generator():
    model = GPT2LMHeadModel.from_pretrained('./startup-pitch-lora')  # Load your fine-tuned model
    tokenizer = GPT2Tokenizer.from_pretrained('./startup-pitch-lora')  # Load the corresponding tokenizer
    return pipeline('text-generation', model=model, tokenizer=tokenizer)

generator = load_generator()

# UI
st.title("🚀 Start-Up Pitch Generator")
st.write("Enter your startup idea and get a short, powerful pitch!")

# User input
idea = st.text_input("Startup Idea", placeholder="e.g., Autonomous sugarcane juice kiosks")

# Button action
if st.button("Generate Pitch"):
    if idea.strip() == "":
        st.warning("Please enter a startup idea.")
    else:
        # Pattern-based prompt for GPT-2
        prompt = (
            "Startup Idea: Autonomous sugarcane juice kiosks\n"
            "Pitch: Imagine a world where fresh sugarcane juice is available 24/7 through AI-powered kiosks. Our autonomous machines bring hygiene, convenience, and nostalgia to every sip.\n\n"
            f"Startup Idea: {idea}\n"
            "Pitch:"
        )

        try:
            output = generator(
                prompt,
                max_length=200,
                num_return_sequences=1,
                pad_token_id=50256,
                do_sample=True,
                temperature=0.9,
                top_p=0.95
            )

            generated_text = output[0]["generated_text"]
            pitch = generated_text.replace(prompt, "").strip().split("\n")[0]

            # Clean up pitch
            pitch = pitch.strip("1234567890).•- ")

            st.success("🎯 Generated Pitch:")
            st.markdown(f"> 💡 *{pitch}*")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
