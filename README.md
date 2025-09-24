1. Download PDFs from S3 and extract them into ~/Downloads/domaindata or another folder if you prefer so
2. Download pdf_reader notebook and run its cells. You need to have postgres and pgvector preinstalled with a user, password and database. The notebook creates a table called document_chunks that contains all PDFs split into overlapping chunks, together with each chunk's embeddings. 
3. Clone the repo.
4. If your IDE does not create virtual environment by default, create it manually. Once you activate the venv, install CPU only versions of pytorch with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu.
5. Run pip install -r requirements.txt
6. Create an .env file in the root folder with your DB credentials.
7. Install TinyLLama with the following script in the terminal: python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
8. Start the server with uvicorn main:app
9. Open rag_frontend.html and test
Some suggested prompts that match documents in the knowledge base:
"Is ethanol a viable fuel in aviation?"
"cotton market outlook"
"Do workers receive their salaries regularly in Latin America or there are delays in payments?"
