from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, StringField, SubmitField
from wtforms.validators import DataRequired
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

class QueryForm(FlaskForm):
    pdf = FileField('Upload your PDF', validators=[DataRequired()])
    question = StringField('Ask a question about your PDF:', validators=[DataRequired()])
    submit = SubmitField('Submit')

def process_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Add scikit-learn TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)

    # Add TensorFlow/Keras model
    model = load_model('your_keras_model.h5')  # Load your Keras model
    embeddings = model.predict(tfidf_matrix)
    knowledge_base = FAISS(embeddings)

    return knowledge_base

@app.route('/', methods=['GET', 'POST'])
def ask_pdf():
    form = QueryForm()

    if form.validate_on_submit():
        pdf = request.files['pdf']
        user_question = form.question.data

        if pdf and user_question:
            knowledge_base = process_pdf(pdf)

            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            return render_template('result.html', response=response)

    return render_template('index.html', form=form)

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True)
