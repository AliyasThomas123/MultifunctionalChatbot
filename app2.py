import streamlit as st
from huggingface_hub import InferenceClient
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForQuestionAnswering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load Hugging Face client for image generation
client = InferenceClient(model="stabilityai/stable-diffusion-3.5-large", token="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")

# Function Definitions
def predict_next_word(input_text, num_words=20):
    try:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_new_tokens=num_words,
            do_sample=True,
            temperature=0.7
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        new_words = generated_text[len(input_text):].strip()
        return new_words
    except Exception as e:
        return f"Error: {str(e)}"

def generate_image_hf(prompt):
    try:
        image = client.text_to_image(prompt)
        image_stream = BytesIO()
        image.save(image_stream, format="PNG")
        image_stream.seek(0)
        return image_stream
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentiment_analysis_model():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_question_answering_model():
    return pipeline("question-answering")

# Application Frontend
st.title("MultiFunctional ChatBot")
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
    actual_next_word = st.text_input("Enter the actual next word (for evaluation):")
    if st.button("Predict"):
        if prompt:
            predicted_words = predict_next_word(prompt)
            st.success(f"Predicted: {predicted_words}")
            if actual_next_word:
                predicted_word_list = predicted_words.split()
                accuracy = accuracy_score([actual_next_word], predicted_word_list[:1])
                st.write(f"Accuracy: {accuracy:.2f}")
        else:
            st.error("Please enter a prompt.")

elif task == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    text = st.text_area("Enter the text to analyze:")
    actual_sentiment = st.selectbox("Enter the actual sentiment:", ["Positive", "Negative", "Neutral"])
    if st.button("Analyze"):
        if text:
            sentiment_analyzer = load_sentiment_analysis_model()
            result = sentiment_analyzer(text)
            st.success(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.2f}")
            if actual_sentiment:
                predicted = result[0]["label"].lower()
                actual = actual_sentiment.lower()
                accuracy = accuracy_score([actual], [predicted])
                precision = precision_score([actual], [predicted], average='macro', zero_division=0)
                recall = recall_score([actual], [predicted], average='macro', zero_division=0)
                f1 = f1_score([actual], [predicted], average='macro', zero_division=0)
                st.write(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
        else:
            st.error("Please enter some text to analyze.")

elif task == "Question Answering":
    st.header("Question Answering")
    context = st.text_area("Enter the context:")
    question = st.text_input("Enter the question:")
    #actual_answer = st.text_input("Enter the actual answer (for evaluation):")
    if st.button("Answer"):
        if context and question:
            qa_model = load_question_answering_model()
            answer = qa_model(question=question, context=context)
            st.success(f"Answer: {answer['answer']}")
            #if actual_answer:
                #predicted = answer["answer"]
                #accuracy = accuracy_score([actual_answer], [predicted])
                #st.write(f"Accuracy: {accuracy:.2f}")
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
