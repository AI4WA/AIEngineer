import os
from typing import List, Dict
from pathlib import Path
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import uuid
import loguru

logger = loguru.logger


class PDFQuestionAnsweringSystem:
    def __init__(self, pdf_dir: str, output_dir: str):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir

        # Initialize embedding model
        self.embedding_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large")
        self.embedding_model = AutoModel.from_pretrained("intfloat/e5-large")

        self.qdrant_client = QdrantClient(":memory:")  # In-memory Qdrant instance

        # Initialize LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        self.llm_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                                              trust_remote_code=True)

    def get_embedding(self, text: str) -> List[float]:
        # Prefix for e5 models
        text = f"passage: {text}"
        inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings

    def process_pdfs(self):
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith(".pdf"):
                # logger.info(f"Processing {filename}")
                self.process_single_pdf(os.path.join(self.pdf_dir, filename))

    def process_single_pdf(self, pdf_path: str):
        output_dir = os.path.join(self.output_dir, os.path.basename(pdf_path).replace(".pdf", ""))
        os.makedirs(output_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        content_list = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            # logger.info(text)
            # split the text into lines and remove empty lines

            sentences = text.split("\n")
            for sentence in sentences:
                if sentence.strip() == "":
                    continue
                content_list.append({
                    'id': str(uuid.uuid4()),  # Generate UUID here
                    'text': sentence,
                    'page_num': page_num,
                    'pdf_name': os.path.basename(pdf_path)
                })

            # Save images (optional)
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"{output_dir}/image_{page_num}_{xref}.{image_ext}"
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)

        # Save as markdown
        md_content = "\n\n".join([f"## Page {item['page_num'] + 1}\n\n{item['text']}" for item in content_list])
        with open(f"{output_dir}/{os.path.basename(pdf_path)}.md", "w", encoding="utf-8") as md_file:
            md_file.write(md_content)

        self.embed_and_index(content_list)

    def embed_and_index(self, content_list: List[Dict]):
        points = []
        for item in content_list:
            text = item.get('text', '')
            if text:
                embedding = self.get_embedding(text)
                points.append(PointStruct(
                    id=item['id'],  # Use the UUID generated in process_single_pdf
                    vector=embedding,
                    payload={
                        'text': text,
                        'pdf_name': item['pdf_name'],
                        'page_num': item['page_num']
                    }
                ))

        if points:
            self.qdrant_client.upsert(
                collection_name="migration",
                points=points
            )

    def setup_vector_db(self):
        self.qdrant_client.create_collection(
            collection_name="migration",
            vectors_config=VectorParams(size=1024,  # Size for e5-large
                                        distance=Distance.COSINE)
        )

    def answer_question(self, question: str) -> str:
        question_embedding = self.get_embedding(question)
        logger.debug(f"Question embedding: {question_embedding}")
        search_results = self.qdrant_client.search(
            collection_name="migration",
            query_vector=question_embedding,
            limit=10
        )
        logger.debug(f"Search results: {search_results}")

        # Truncate context if it's too long
        max_context_length = 1000  # Adjust this value as needed
        context = ""
        for hit in search_results:
            new_context = f"[PDF: {hit.payload['pdf_name']}, Page: {hit.payload['page_num'] + 1}] {hit.payload['text']}\n"
            if len(context) + len(new_context) > max_context_length:
                context += new_context
                context = context[:max_context_length]
                break
            context += new_context

        prompt = f"Context: {context.strip()}\n\nQuestion: {question}\n\nAnswer:"
        logger.info(f"Prompt: {prompt}")

        input_ids = self.llm_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)

        with torch.no_grad():
            output = self.llm_model.generate(
                input_ids,
                max_new_tokens=512,  # Generate up to 512 new tokens
                num_return_sequences=1,
                pad_token_id=self.llm_tokenizer.eos_token_id  # Ensure proper padding
            )

        answer = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.split("Answer:")[-1].strip()

    def run(self):
        self.setup_vector_db()
        self.process_pdfs()

        while True:
            question = input("Ask a question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            answer = self.answer_question(question)
            print(f"Answer: {answer}")


if __name__ == "__main__":
    pdf_dir = "demo_data/"
    output_dir = "output/"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_system = PDFQuestionAnsweringSystem(pdf_dir, output_dir)
    qa_system.run()
