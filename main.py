__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import re
from dotenv import load_dotenv
from langchain.docstore.document import Document
import streamlit as st
from services.excel_processor import process_uploaded_file
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain
from langchain.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="SearchM CampaignBrief ChatBot", page_icon="💬", layout="wide")
st.title("SearchM CampaignBrief 💬 ChatBot")
st.write("---")
st.markdown("<h2 style='text-align: center; color: grey;'>캠페인 브리프 정보를 제공하는 챗봇입니다</h2>", unsafe_allow_html=True)
st.write("---")
st.header("궁금한 정보를 물어보세요")

if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'file_sheet_data' not in st.session_state:
    st.session_state['file_sheet_data'] = {}
if 'summary' not in st.session_state:
    st.session_state['summary'] = []

uploaded_files = st.file_uploader("캠페인 브리프 파일을 업로드하세요", type=["xlsx"], accept_multiple_files=True)
st.write("---")

save_directory = st.text_input("파일을 저장할 디렉토리를 입력하세요")


def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)

    if uploaded_files:
        combined_response, file_sheet_data = process_uploaded_file(uploaded_files, user_input, save_directory)
        st.session_state.generated.append({'type': 'normal', 'data': combined_response})
        st.session_state.file_sheet_data.update(file_sheet_data)

        # Document 생성
        file_data = []
        total = []
        for file_name, sheets in st.session_state['file_sheet_data'].items():
            file_data.append(file_name)
            total.append([])
            for sheet_name, response in sheets.items():
                total[-1].append([sheet_name, response])

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

        # Document 생성
        docs = []
        for i in range(len(total)):
            temp = file_data[i]
            for j in range(len(total[i])):
                page_content = str(total[i][j][1])
                result = extract_ymd(total[i][j][0])

                doc = Document(
                    page_content=f'{temp}파일의 {result[0]}년 {result[1]}월 {result[2]}일 {total[i][j][0]} 시트에 대한 답변 : {page_content}',
                    metadata={'file_name': temp, 'sheet_name': total[i][j][0],
                              'Year': result[0], 'Month': result[1], 'Day': result[2]}
                )
                docs.append(doc)

        # LLM을 이용한 요약 생성
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        reduce_template = """
            다음은 답변에 대한 요약의 집합입니다:{docs}
            답변 형식은 경어체를 사용하세요.
            
            {docs}의 내용들을 참고하여 조건에 맞는 답변들을 제공해주세요.
            
            만약 월별 자료를 요청할 시 해당 월 자료만 내용에 포함해주고 다른 월 자료는 제공할 필요 없습니다.
            연, 월, 일 중 00 이 있으면 이 부분은 출력하지마세요.
        """

        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="docs"
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4096
        )

        # Update the summary in the session state
        st.session_state['summary'].append(reduce_documents_chain.run(docs))

    else:
        st.session_state.generated.append({'type': 'normal', 'data': "파일을 업로드하지 않았거나, 파일 처리 중 오류가 발생했습니다."})


def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]
    st.session_state.file_sheet_data = {}
    st.session_state.summary = []


with st.container():
    st.text_input("User Input:", on_change=on_input_change, key="user_input")

# 대화 내용 표시 부분
chat_placeholder = st.empty()

with chat_placeholder.container():
    for i in range(len(st.session_state['past'])):
        message(st.session_state['past'][i], is_user=True, key=f"user_{i}")
        message(st.session_state['generated'][i]['data'], key=f"response_{i}")
        if i < len(st.session_state['summary']):
            message(st.session_state['summary'][i], key=f"summary_{i}")

    if len(st.session_state['summary']) > len(st.session_state['past']):
        for i in range(len(st.session_state['past']), len(st.session_state['summary'])):
            message(st.session_state['summary'][i], key=f"summary_{i}")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>© 2024 SearchM. All rights reserved.</p>", unsafe_allow_html=True)
