import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up the Streamlit UI
st.title("Sentiment Analysis Using LangChain")
st.write("===============================[Enter some text below...===============================]")

# Inject CSS to change background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ff0000;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

api_key = st.secrets["gemini"]["api_key"]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

sentiment_prompt = PromptTemplate(
    input_variables=["text"],
    template="Analyze the sentiment of the following text and respond with Positive, Negative, or Neutral:\n\n{text}"
)

threat_prompt = PromptTemplate(
    input_variables=["text"],
    template="Identify any threat-related words in the following sentence. Return only the words that indicate potential harm, violence, or danger.{text}"
)

named_entity_prompt = PromptTemplate(
    input_variables=["text"],
    template="Analyze the given text and identify the names and places, organizations, devices in the text.{text}"
)


sentiment_chain = LLMChain(llm = llm, prompt = sentiment_prompt)
threat_chain =  LLMChain(llm = llm, prompt = threat_prompt)
summary_chain =  LLMChain(llm = llm, prompt = named_entity_prompt)


def analyze_text(text):
    sentiment = sentiment_chain.run(text).strip()
    threats = threat_chain.run(text).strip()
    summary = summary_chain.run(text).strip()

    return {
        "Sentiment": sentiment,
        "Threat Words": threats,
        "Named_entities": summary
    }

# Text input from user
user_input = st.text_area("Your text:", height=100)

# Analyze button
if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Perform sentiment analysis
        #blob = TextBlob(user_input)
        #sentiment = blob.sentiment

        result = analyze_text(user_input)

        # Display results
        st.subheader("Sentiment Analysis, Threat words and Named entities in text")

        st.write(result)






