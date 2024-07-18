from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

def summarize_responses(file_sheet_data, user_input):
    summaries = []

    for file_name, sheets in file_sheet_data.items():
        file_summary = f"<p style='font-size:18px; font-weight:bold;'>파일: {file_name}</p>"
        for sheet_name, response in sheets.items():
            file_summary += f"<p><b>시트: {sheet_name}</b><br>{response}</p>"
        summaries.append(file_summary)

    combined_summary = " ".join(summaries)
    summary_response = generate_summary(combined_summary, user_input)

    return summary_response

def generate_summary(content, user_input):
    system_template = """
        주어진 자료에 대한 요약을 제공해 주세요. 주어진 자료는 다음과 같은 형식을 따릅니다:
        {file_name}: 파일명
        {sheet_name}: 시트명
        {content_answer}: 시트 내용 요약

        파일 및 시트별로 주어진 내용을 간단히 요약해 주세요. 내용이 같으면 한 번에 출력합니다.
        또한, 사용자가 제공한 추가 입력을 반영하여 가능한 답변을 제공해 주세요.
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{content}\n\n사용자 입력: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = LLMChain(llm=llm, prompt=chat_prompt)

    response = chain.run({"content": content, "user_input": user_input})
    return response
