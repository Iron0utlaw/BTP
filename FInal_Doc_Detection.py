import os
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinter.ttk import Progressbar
import string
from sentence_transformers import SentenceTransformer, util
import torch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import nltk
import pytesseract
from pdf2image import convert_from_path
import PyPDF2

# Ensure required NLTK data is downloaded
nltk.download('punkt_tab')
nltk.download('stopwords')

# Preprocessing function to clean and tokenize text
def preprocess_text(text):
    """
    Preprocesses the input text by tokenizing it into sentences, converting to lowercase, removing punctuation, and filtering out stopwords.

    Args:
        text (str): The input text to preprocess.

    Returns:
        list: A list of preprocessed sentences.
    """
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)

    sentences = sent_tokenize(text)
    preprocessed_sentences = []

    for sentence in sentences:
        sentence = sentence.lower().translate(translator)  # Lowercase and remove punctuation
        words = [word for word in sentence.split() if word not in stop_words]
        preprocessed_sentences.append(" ".join(words))

    return preprocessed_sentences

# Function to calculate similarity using embeddings
def calculate_similarity(text1, text2):
    """
    Calculates the similarity between two texts using SentenceTransformer embeddings and cosine similarity.

    Args:
        text1 (str): The first text document.
        text2 (str): The second text document.

    Returns:
        tuple: A tuple containing the average similarity score, cosine similarity matrix, and preprocessed sentences from both texts.
    """
    model = SentenceTransformer('paraphrase-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

    sentences1 = preprocess_text(text1)
    sentences2 = preprocess_text(text2)

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
    max_similarities = torch.max(cosine_similarities, dim=1).values
    avg_similarity = torch.mean(max_similarities).item()

    return avg_similarity, cosine_similarities, sentences1, sentences2

# Function to highlight plagiarized content
def highlight_plagiarism(cosine_similarities, sentences_to_highlight, sentences_to_compare):
    """
    Highlights plagiarized sentences based on cosine similarity scores.

    Args:
        cosine_similarities (torch.Tensor): Cosine similarity matrix between sentences of two documents.
        sentences_to_highlight (list): Sentences to be checked for plagiarism.
        sentences_to_compare (list): Sentences to compare against.

    Returns:
        str: Highlighted text with tags indicating high or medium similarity.
    """
    highlighted_text = ""

    for i, sentence in enumerate(sentences_to_highlight):
        if i < cosine_similarities.size(0):  # Ensure i is within bounds for cosine_similarities
            max_similarity = torch.max(cosine_similarities[i]).item()

            if max_similarity >= 0.9:  # High similarity
                highlighted_text += f"[RED] {sentence.strip()} [END]\n"
            elif 0.75 <= max_similarity < 0.9:  # Medium similarity
                highlighted_text += f"[YELLOW] {sentence.strip()} [END]\n"
            else:
                highlighted_text += f"{sentence.strip()}\n"

    return highlighted_text

# Function to extract text from PDFs (both text-based and image-based using OCR)
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file, using PyPDF2 for text-based PDFs and Tesseract OCR for image-based PDFs.

    Args:
        pdf_path (str): The file path of the PDF.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""

        if not text.strip():
            images = convert_from_path(pdf_path)  # Convert pages to images
            for image in images:
                text += pytesseract.image_to_string(image)  # OCR to extract text from images
    except Exception as e:
        messagebox.showerror("Error", f"Error extracting text from PDF: {e}")

    return text

# Function to detect plagiarism and process the selected files
def detect_plagiarism():
    """
    Handles the process of file selection, text extraction, and similarity calculation to detect plagiarism.
    Results are displayed in the GUI with highlighted text.
    """
    def process_similarity():
        loader_label.pack(pady=10)
        progress_bar.pack(pady=5)
        progress_bar.start()

        filepaths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf")])

        if not filepaths:
            loader_label.pack_forget()
            progress_bar.pack_forget()
            progress_bar.stop()
            messagebox.showerror("Error", "No files selected!")
            return

        texts = []
        filenames = []
        for filepath in filepaths:
            filenames.append(os.path.basename(filepath))

            if filepath.lower().endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()

            texts.append(text)

        results = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity_score, cosine_similarities, sentences1, sentences2 = calculate_similarity(texts[i], texts[j])
                highlighted_text = highlight_plagiarism(cosine_similarities, sentences2, sentences1)

                results.append((f"{filenames[i]} vs {filenames[j]}", similarity_score * 100, highlighted_text))

        text_area.delete("1.0", tk.END)
        for result in results:
            text_area.insert(tk.END, f"Comparison: {result[0]}\n")
            text_area.insert(tk.END, f"Similarity Score: {result[1]:.2f}%\n")
            text_area.insert(tk.END, "Highlighted Text:\n")

            for line in result[2].split('\n'):
                if line.startswith("[RED]"):
                    text_area.insert(tk.END, line[6:-5], 'highlighted_high')
                elif line.startswith("[YELLOW]"):
                    text_area.insert(tk.END, line[9:-5], 'highlighted_medium')
                else:
                    text_area.insert(tk.END, line)
                text_area.insert(tk.END, '\n')

        loader_label.pack_forget()
        progress_bar.pack_forget()
        progress_bar.stop()

    threading.Thread(target=process_similarity).start()

# Create the GUI
window = tk.Tk()
window.title("Plagiarism Detector for Multiple Documents")

text_area_label = tk.Label(window, text="Plagiarism Detection Results:")
text_area_label.pack(pady=5)

text_area = scrolledtext.ScrolledText(window, width=80, height=20)
text_area.pack(pady=5)

text_area.tag_configure('highlighted_high', background='red', foreground='white')
text_area.tag_configure('highlighted_medium', background='yellow', foreground='black')

loader_label = tk.Label(window, text="Processing...", font=("Arial", 12))
progress_bar = Progressbar(window, mode='indeterminate')

detect_button = tk.Button(window, text="Select Files and Detect Plagiarism", 
                        command=detect_plagiarism)
detect_button.pack(pady=10)

window.mainloop()
