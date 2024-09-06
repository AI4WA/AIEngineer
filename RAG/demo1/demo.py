# crawler.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from pathlib import Path
import loguru

logger = loguru.logger


class Crawler:
    def __init__(self, start_url: str, output_dir: Path):
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.output_dir = output_dir
        self.visited = set()

    def crawl(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._crawl_recursive(self.start_url)

    def _crawl_recursive(self, url: str):
        if url in self.visited:
            return
        self.visited.add(url)

        try:
            response = requests.get(url)
            if not response.headers.get('Content-Type', '').startswith('text/html'):
                return

            soup = BeautifulSoup(response.text, 'html.parser')
            self._save_content(url, soup.get_text())

            for link in soup.find_all('a', href=True):
                next_url = urljoin(url, link['href'])
                if self._should_crawl(next_url):
                    self._crawl_recursive(next_url)

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")

    def _should_crawl(self, url: str) -> bool:
        return urlparse(url).netloc == self.domain and url not in self.visited

    def _save_content(self, url: str, content: str):
        file_path = self.output_dir / f"{urlparse(url).path.strip('/')}.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        logger.info(f"Saved content from {url} to {file_path}")


# embedding.py
import pandas as pd
from pathlib import Path
import tiktoken
from openai import OpenAI
import numpy as np


class Embedder:
    def __init__(self, input_dir: Path, output_dir: Path, openai_client: OpenAI):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.openai_client = openai_client
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 500

    def process(self):
        df = self._load_and_process_text()
        df = self._split_text(df)
        df = self._get_embeddings(df)
        self._save_embeddings(df)
        return df

    def _load_and_process_text(self) -> pd.DataFrame:
        texts = []
        for file in self.input_dir.glob('*.txt'):
            text = file.read_text()
            texts.append((file.stem, text))

        df = pd.DataFrame(texts, columns=['title', 'text'])
        df['text'] = df['title'] + ". " + df['text'].str.replace('\s+', ' ', regex=True)
        df['n_tokens'] = df['text'].apply(lambda x: len(self.tokenizer.encode(x)))
        return df

    def _split_text(self, df: pd.DataFrame) -> pd.DataFrame:
        shortened = []
        for _, row in df.iterrows():
            if row['n_tokens'] > self.max_tokens:
                shortened.extend(self._split_into_many(row['text']))
            else:
                shortened.append(row['text'])

        new_df = pd.DataFrame(shortened, columns=['text'])
        new_df['n_tokens'] = new_df['text'].apply(lambda x: len(self.tokenizer.encode(x)))
        return new_df

    def _split_into_many(self, text: str) -> list:
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(" " + sentence))
            if current_token_count + sentence_tokens > self.max_tokens:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = []
                current_token_count = 0

            current_chunk.append(sentence)
            current_token_count += sentence_tokens + 1

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    def _get_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        df['embedding'] = df['text'].apply(
            lambda x: self.openai_client.embeddings.create(input=x, model="text-embedding-ada-002").data[0].embedding)
        return df

    def _save_embeddings(self, df: pd.DataFrame):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_dir / 'embeddings.parquet', engine='pyarrow')


# qa.py
from openai import OpenAI
import pandas as pd
import numpy as np
from pathlib import Path


class QASystem:
    def __init__(self, embeddings_path: Path, openai_client: OpenAI):
        self.df = pd.read_parquet(embeddings_path, engine='pyarrow')
        self.openai_client = openai_client

    def answer_question(self, question: str, model: str = "gpt-3.5-turbo", max_tokens: int = 150) -> str:
        context = self._create_context(question)
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant. Answer the question based on the context provided."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
        ]

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    def _create_context(self, question: str, max_len: int = 1800) -> str:
        q_embedding = self.openai_client.embeddings.create(input=question, model="text-embedding-ada-002").data[
            0].embedding
        self.df['distances'] = self.df['embedding'].apply(
            lambda x: np.dot(x, q_embedding) / (np.linalg.norm(x) * np.linalg.norm(q_embedding)))

        returns = []
        cur_len = 0

        for _, row in self.df.sort_values('distances', ascending=False).iterrows():
            cur_len += row['n_tokens']
            if cur_len > max_len:
                break
            returns.append(row['text'])

        return "\n\n###\n\n".join(returns)


# main.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
START_URL = "https://nlp-tlp.org/"
DATA_DIR = Path("data")
CRAWL_DIR = DATA_DIR / "crawled"
PROCESSED_DIR = DATA_DIR / "processed"


def main():
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Step 1: Crawl the website
    crawler = Crawler(START_URL, CRAWL_DIR)
    crawler.crawl()

    # Step 2: Process and embed the text
    embedder = Embedder(CRAWL_DIR, PROCESSED_DIR, openai_client)
    embedder.process()

    # Step 3: Set up QA system
    qa_system = QASystem(PROCESSED_DIR / 'embeddings.parquet', openai_client)

    # Step 4: Answer questions
    while True:
        question = input("Ask a question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        answer = qa_system.answer_question(question)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
