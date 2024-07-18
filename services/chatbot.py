from services.model_setup import setup_model_and_generate_response

def generate_responses(processed_documents, user_input, output_file_path):
    doc_chunks = create_document_chunks(processed_documents, output_file_path)
    response = setup_model_and_generate_response(doc_chunks, user_input)
    return response

def create_document_chunks(processed_documents, output_file_path):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document

    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, separators=["$$"], chunk_overlap=50)

    for document in processed_documents:
        chunks = text_splitter.split_text(document)
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk, metadata={"page": i, "source": output_file_path})
            doc_chunks.append(doc)

    return doc_chunks