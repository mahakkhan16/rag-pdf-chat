from rag_engine import process_pdf, answer_question

pdf_path = "CampusAI_IEEE_Research_Paper.pdf"  # must match exactly, including the space

print("Processing PDF...")
vectorstore = process_pdf(pdf_path)

question = "What is the main goal or purpose of this research?"
result = answer_question(vectorstore, question)

print("\nAnswer:", result["answer"])
print("\nNumber of source chunks used:", len(result["sources"]))