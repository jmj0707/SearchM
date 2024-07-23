import re
import os
import pandas as pd
from langchain.docstore.document import Document
from services.chatbot import generate_responses

def process_uploaded_file(uploaded_files, user_input, save_directory):
    sheet_responses = []
    response_generated = False
    file_sheet_data = {}

    def extract_ymd(sheet_name):
        patterns = [
            (r'(\d{2})(\d{2})(\d{2})', lambda m: (m.group(1), m.group(2), m.group(3))),  # YYMMDD
            (r'(\d{2})년(\d{2})월(\d{2})일', lambda m: (m.group(1), m.group(2), m.group(3))),  # YY년MM월DD일
            (r'(\d{4})년(\d{2})월(\d{2})일', lambda m: (m.group(1), m.group(2), m.group(3))),  # YYYY년MM월DD일
            (r'(\d{2})/(\d{2})/(\d{2})', lambda m: (m.group(1), m.group(2), m.group(3))),  # YY/MM/DD
            (r'(\d{2})(\d{2})', lambda m: ('00', m.group(1), m.group(2))),  # MMDD
            (r'(\d{2})/(\d{2})', lambda m: ('00', m.group(1), m.group(2))),  # MM/DD
            (r'(\d{2})월 (\d{2})일', lambda m: ('00', m.group(1), m.group(2))),  # MM월 DD일
            (r'(\d{2})월', lambda m: ('00', m.group(1), '00')),  # MM월
            (r'(\d{2})일', lambda m: ('00', '00', m.group(1)))  # DD일
        ]
        for pattern, func in patterns:
            match = re.search(pattern, sheet_name)
            if match:
                return func(match)
        return ("00", "00", "00")  # 날짜 형식이 없으면 00/00으로

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
                    result = extract_ymd(sheet_name)
                    output_file_path = os.path.join(save_path, f'{sheet_name}.xlsx')
                    data.to_excel(output_file_path, index=False)

                    text = data.to_string(index=False)

                    document = Document(
                        metadata={'source': output_file_path, 'sheet': sheet_name,
                                  'Year': result[0], 'Month': result[1], 'Day': result[2]
                        },
                        page_content=text
                    )

                    processed_documents = []

                    keywords = [
                        "1. Client 광고주", "2. Brand", "3. Date", "4. 광고제품", "5. 캠페인", "5-1. 캠페인",
                        "5-2. 광고집행", "6. 프로모션", "7. 광고목적", "8. 광고", "9. 시장세분화", "10. 타겟팅", "11. 포지셔닝"
                    ]

                    text = document.page_content
                    text = re.sub(r'\bNaN\b', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    text = re.sub(r'\n{2,}', '\n', text)
                    for keyword in keywords:
                        pattern = re.escape(keyword)
                        text = re.sub(pattern, r'$$' + keyword, text)
                    pattern2 = r"Unnamed: [0-9]"
                    text = re.sub(pattern2, '', text)
                    text = re.sub(r'12\. 시안 제작.*', '', text, flags=re.DOTALL)

                    # print(text)
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

