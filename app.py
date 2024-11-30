import streamlit as st
from huggingface_hub import InferenceClient
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"  # You can replace this with "EleutherAI/gpt-neo-125M" or other Hugging Face models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification
)
from PIL import Image
import openai
client = InferenceClient(model="stabilityai/stable-diffusion-3.5-large", token="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")

def predict_next_word(input_text, num_words=20):
    try:
        # Tokenize the input
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate the next tokens
        output = model.generate(
            input_ids,
            max_new_tokens=num_words,  # Number of tokens to predict
            do_sample=True,           # Enable sampling to get varied results
            temperature=0.7           # Adjust creativity (lower is less creative, higher is more)
        )

        # Decode the generated tokens
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Return only the newly generated words (remove the input text)
        new_words = generated_text[len(input_text):].strip()
        return new_words
    except Exception as e:
        return f"Error: {str(e)}"

def generate_image_hf(prompt):
    try:
        # Generate the image
        image = client.text_to_image(prompt)  # Returns a PIL image object
        # Convert the image to an in-memory BytesIO object
        image_stream = BytesIO()
        image.save(image_stream, format="PNG")
        image_stream.seek(0)  # Reset the pointer to the start of the stream
        return image_stream
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None
# Hugging Face Pretrained Models
@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_next_word_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

@st.cache_resource
def load_sentiment_analysis_model():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_question_answering_model():
    return pipeline("question-answering")

@st.cache_resource
def load_openai_api_key():
    # Replace this with a secure method of fetching your API key
    return "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"

# Application Frontend
st.title("AI-Powered Application with Multiple Tasks")
st.sidebar.title("Task Selector")
task = st.sidebar.selectbox(
    "Choose a Task:",
    [
        "Text Summarization",
        "Next Word Prediction",
        "Sentiment Analysis",
        "Question Answering",
        "Image Generation",
    ]
)

# Task Implementations
if task == "Text Summarization":
    st.header("Text Summarization")
    text = st.text_area("Enter the text to summarize:")
    if st.button("Summarize"):
        if text:
            summarizer = load_summarization_model()
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
            st.success(summary[0]["summary_text"])
        else:
            st.error("Please enter some text to summarize.")

elif task == "Next Word Prediction":
    st.header("Next Word Prediction")
    prompt = st.text_input("Enter a prompt:")
    if st.button("Predict"):
        if prompt:
           res =  predict_next_word(prompt)
           
           st.success(res)
        else:
            st.error("Please enter a prompt.")

elif task == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    text = st.text_area("Enter the text to analyze:")
    if st.button("Analyze"):
        if text:
            sentiment_analyzer = load_sentiment_analysis_model()
            result = sentiment_analyzer(text)
            st.success(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.2f}")
        else:
            st.error("Please enter some text to analyze.")

elif task == "Question Answering":
    st.header("Question Answering")
    context = st.text_area("Enter the context:")
    question = st.text_input("Enter the question:")
    if st.button("Answer"):
        if context and question:
            qa_model = load_question_answering_model()
            answer = qa_model(question=question, context=context)
            st.success(answer["answer"])
        else:
            st.error("Please enter both context and question.")

elif task == "Image Generation":
    st.header("Image Generation")
    prompt = st.text_input("Enter a description for the image:")
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating image..."):
                image_stream = generate_image_hf(prompt)
            if image_stream:
                st.image(image_stream, caption="Generated Image", use_column_width=True)
        else:
            st.warning("Please enter a prompt to generate an image.")

# Metrics Section
st.sidebar.title("Evaluation Metrics")
st.sidebar.write("""
- **Accuracy**: How correct the model outputs are.
- **Precision**: Percentage of true positives.
- **Recall**: Ability to find all true positives.
- **F1-Score**: Balance of Precision and Recall.
- **User Satisfaction**: Gathered via feedback.
""")
