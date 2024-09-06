import os
from typing import List, Dict
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from pathlib import Path
import torch
import magic_pdf.model as model_config

model_config.__use_inside_model__ = True


class PDFQuestionAnsweringSystem:
    def __init__(self, pdf_dir: str, output_dir: str):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.embedding_model = SentenceTransformer('dunzhang/stella_en_400M_v5', trust_remote_code=True)
        self.qdrant_client = QdrantClient(":memory:")  # In-memory Qdrant instance
        rope_scaling = {"type": "yarn"}  # or {"type": "su"}
        config = AutoConfig.from_pretrained("microsoft/Phi-3.5-mini-instruct", rope_scaling="yarn")
        self.llm_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        self.llm_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", config=config)

    def process_pdfs(self):
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith(".pdf"):
                self.process_single_pdf(os.path.join(self.pdf_dir, filename))

    def process_single_pdf(self, pdf_path: str):
        pdf_bytes = open(pdf_path, "rb").read()
        output_dir = os.path.join(self.output_dir, os.path.basename(pdf_path).replace(".pdf", ""))
        os.makedirs(output_dir, exist_ok=True)

        image_writer = DiskReaderWriter(os.path.join(output_dir, "images"))
        md_writer = DiskReaderWriter(output_dir)
        model_json = []
        jso_useful_key = {"_pdf_type": "", "model_list": model_json}
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()

        content_list = pipe.pipe_mk_uni_format(output_dir, drop_mode="none")
        md_content = pipe.pipe_mk_markdown(output_dir, drop_mode="none")

        md_writer.write(content=md_content, path=f"{os.path.basename(pdf_path)}.md")

        self.embed_and_index(content_list, os.path.basename(pdf_path))

    def embed_and_index(self, content_list: List[Dict], pdf_name: str):
        for item in content_list:
            text = item.get('text', '')
            if text:
                embedding = self.embedding_model.encode(text)
                self.qdrant_client.upsert(
                    collection_name="pdf_content",
                    points=[{
                        'id': item['id'],
                        'payload': {'text': text, 'pdf_name': pdf_name},
                        'vector': embedding.tolist()
                    }]
                )

    def setup_vector_db(self):
        self.qdrant_client.create_collection(
            collection_name="pdf_content",
            vectors_config=VectorParams(size=self.embedding_model.get_sentence_embedding_dimension(),
                                        distance=Distance.COSINE)
        )

    def answer_question(self, question: str) -> str:
        question_embedding = self.embedding_model.encode(question)
        search_results = self.qdrant_client.search(
            collection_name="pdf_content",
            query_vector=question_embedding.tolist(),
            limit=3
        )

        context = "\n".join([hit.payload['text'] for hit in search_results])
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        input_ids = self.llm_tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.llm_model.generate(input_ids, max_length=512, num_return_sequences=1)

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
