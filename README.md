1. Download PDFs from S3 and extract them into ~/Downloads/domaindata or another folder if you prefer so
2. Download pdf_reader notebook and run its cells. You need to have postgres and pgvector preinstalled. The notebook creates a table called document_chunks that holds all the PDFs into chunks with each chunk embeddings
3. Clone the repo and install its requirements. Create an .env file in the root folder with you DB credentials.
4. Install TinyLLama with the following script in the terminal: python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
5. Start the server with uvicorn main:app --reload
