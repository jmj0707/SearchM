import re
import os
import pandas as pd
from langchain.docstore.document import Document
from services.chatbot import generate_responses

def process_uploaded_file(uploaded_files, user_input, save_directory):
    sheet_responses = []
    response_generated = False
    file_sheet_data = {}

    for uploaded_file in uploaded_files:
        try:
            file_path = uploaded_file.name
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join(save_directory, file_name)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            excel_data = pd.read_excel(uploaded_file, sheet_name=None)
            file_sheet_data[file_name] = {}

            for sheet_name, data in excel_data.items():
                if sheet_name != 'Campaign Brief':
                    output_file_path = os.path.join(save_path, f'{sheet_name}.xlsx')
                    data.to_excel(output_file_path, index=False)

                    text = data.to_string(index=False)

                    document = Document(
                        metadata={'source': output_file_path, 'sheet': sheet_name},
                        page_content=text
                    )

                    processed_documents = []
                    text = document.page_content
                    text = re.sub(r'\bNaN\b', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r'\n{2,}', '\n', text)
                    text = re.sub(r'\\n', '\n', text)
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r'Unnamed: \d+', '', text)
                    text = re.sub(r'12\. 시안 제작.*', '', text, flags=re.DOTALL)
                    text = re.sub(r'(\d+\.\s*|\d+-\d+\.\s*)', r'$$\1', text)
                    processed_documents.append(text)

                    response = generate_responses(processed_documents, user_input, output_file_path)

                    file_sheet_data[file_name][sheet_name] = response
                    sheet_responses.append(
                        f"<p style='font-size:18px; font-weight:bold;'> {file_name} 파일의 {sheet_name} 시트 정보</p>\n"
                        f"{response if response else '결과를 가져올 수 없습니다.'}"
                    )
                    response_generated = True

        except Exception as e:
            sheet_responses.append(f"오류 발생: {e}")

    if not response_generated:
        sheet_responses.append("파일을 업로드하지 않았거나, 파일 처리 중 오류가 발생했습니다.")

    return "\n\n---\n\n".join(sheet_responses), file_sheet_data
