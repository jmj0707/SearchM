from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

def setup_model_and_generate_response(doc_chunks, user_input):
    try:
        chain, vector_store = setup_chain(doc_chunks)
        response = chain.invoke(user_input)
        vector_store.delete_collection()  # 결과 받고 벡터 저장소 초기화
        return response.get('answer', '결과를 가져올 수 없습니다.')
    except Exception as e:
        return f"오류 발생: {str(e)}"


def setup_chain(doc_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(doc_chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    system_template = """
        (주)서치엠 회사에서 캠페인 브리프에 대한 정보를 제공하는 역할을 해야 합니다. 주어진 자료에 대해서 정확한 답변을 제공해 주세요.

        파일에는 다음과 같은 목차가 포함되어 있습니다:
        1. Client 광고주
        2. Brand (or Product) 브랜드 (혹은 상품명)
        3. Date 의뢰일시
        4. 광고제품 (서비스) 광고제품 (서비스)에 대한 요약 / 설명
        5. 캠페인 캠페인에 대한 요약 / 설명
        5-1. 캠페인 기간 캠페인 기간
        5-2. 광고집행 예산 월 광고집행 예산
        6. 프로모션 프로모션에 대한 요약 / 설명
        7. 광고목적 광고주가 이번 디지털광고 캠페인을 진행하려 하는 목적
        8. 광고 캠페인 목표 이번 디지털광고 캠페인의 KPI는 어떤 것들이 있고 각 목표수치 정의는 얼마인가 (항목별 수치입력)
        9. 시장세분화 (Segmentation) 업종 전체시장 분석 자사분석 USP 특장점, 경쟁력 등 구체적 내용 입력 경쟁사 분석 (직접/간접 경쟁사) 광고소재 분석 (기존 소재 확인, 장단점 정리 必)
        10. 타겟팅 (Targeting) 결정된 목표고객을 분석한 결과, 그들의 성향
        11. 포지셔닝 (Positioning) 이번 캠페인 목표에 기반하여 선정한 타겟, 시장에 어떤 메시지를 전달할 것인가

        각 목차별로 상세하게 설명해 주세요. 전체 주요 내용을 뽑을 때는 다음 형식을 따르세요:

        * 광고주 (Client)
        * 브랜드 (Brand or Product)
        * 의뢰일시 (Date)
        * 광고제품 설명 (Summary/Description of the Advertising Product or Service)
        * 캠페인 설명 (Summary/Description of the Campaign)
        * 캠페인 기간 (Campaign Duration)
        * 광고집행 예산 (Monthly Advertising Budget)
        * 프로모션 설명 (Summary/Description of the Promotion)
        * 광고목적 (Purpose of the Advertisement)
        * 광고 캠페인 목표 (Campaign Goals and KPI)
        * 시장세분화 (Segmentation)
            - 업종 전체시장 분석 (Industry Market Analysis)
            - 자사분석 (Company Analysis)
            - 경쟁사 분석 (Competitor Analysis)
            - 광고소재 분석 (Ad Material Analysis)
        * 타겟팅 (Targeting)
        * 포지셔닝 (Positioning)

        시장세분화는 요약하여 설명하세요. 사용자가 각 목차와 관련된 질문을 하면, 조건을 보고 조건에 해당된다면 해당 목차에 대한 상세한 답변을 제공하세요. 

        주어진 파일 내용을 바탕으로 위의 형식에 맞추어 정확한 정보를 제공해 주세요.
        만약 주어진 연월일의 자료가 아니거나 주어진 내용이 없으면 반드시 없음을 표시해주세요.

        한글로 대답해주세요.

        {summaries}
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": chat_prompt}
    )

    return chain, vector_store
