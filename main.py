from dotenv import load_dotenv
from langchain.docstore.document import Document
import streamlit as st
from services.excel_processor import process_uploaded_file
from streamlit_chat import message

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
    else:
        st.session_state.generated.append({'type': 'normal', 'data': "파일을 업로드하지 않았거나, 파일 처리 중 오류가 발생했습니다."})

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]
    st.session_state.file_sheet_data = {}

chat_placeholder = st.empty()

with chat_placeholder.container():
    if len(st.session_state['past']) != len(st.session_state['generated']):
        st.warning("데이터 길이 불일치: past와 generated의 길이가 다릅니다.")

    for i in range(min(len(st.session_state['past']), len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(
            st.session_state['generated'][i]['data'],
            key=f"{i}",
            allow_html=True,
            is_table=True if st.session_state['generated'][i].get('type') == 'table' else False
        )

    st.button("Clear message", on_click=on_btn_click)

with st.container():
    st.text_input("User Input:", on_change=on_input_change, key="user_input")

file_data = []
total = []

st.write("### 업로드된 파일 및 시트 데이터")
for file_name, sheets in st.session_state['file_sheet_data'].items():
    file_data.append(file_name)
    total.append([])
    for sheet_name, response in sheets.items():
        total[-1].append([sheet_name, response])

docs = []
k = 0
for i in range(len(total)):
    temp = file_data[i]
    for j in range(len(total[i])):
        page_content = str(total[i][j][1])
        k += 1

        doc = Document(
            page_content=page_content,
            metadata={'file_name': temp, 'sheet_name': total[i][j][0], 'page': k}
        )
        docs.append(doc)

for doc in docs:
    st.write(doc.page_content, doc.metadata)

st.write(docs)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>© 2024 SearchM. All rights reserved.</p>", unsafe_allow_html=True)
