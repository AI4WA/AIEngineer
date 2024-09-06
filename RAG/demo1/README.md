# RAG Demo 1 with a website

Website is: https://nlp-tlp.org

## Setup

1. Create the python environment

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

2. Download the code or copy the code from the `demo.py` file, paste it in your python file.
3. Setup the OpenAI API key, create one here: https://platform.openai.com/api-keys. Then create a .env file and add the
   key to it.

```
OPENAI_API_KEY=your_key_here
```

4. Run the code

```bash
python demo.py
```
