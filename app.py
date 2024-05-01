import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import hashlib

# (deprecatd) from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

# (deprecatd) from langchain.vectorstores import FAISS # 문서검색을 담당하는 페이스북이 만든 고속 Vector DB. 로컬에 설치하며, 프로그램 종료시 DB는 삭제됨
from langchain_community.vectorstores import FAISS

from htmlTemplates import css, bot_template, user_template   # htmlTemplates.py 파일안에 있는 모듈들을 가져옴

##############################
#             DB             #
##############################
# pdf 로딩 : pdf_docs = st.file_uploader()
# 이후 3단계 : pdf_docs --(1)--> text --(2)-->  chunk --(3)--> vectorstore
# (1) .extract_texts
# (2) CTS
# (3) FAISS.from_texts

def get_pdf_text(pdf_docs):
    # 빈 text 배열 생성
    text = ""
    for pdf in pdf_docs:
        # 페이지별로 배열을 리턴해주는 PdfReader클래스의 객체 생성
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            # 각 페이지의 텍스트를 추출하여 모든 text를 배열로 저장 --> extract_text 매소드를 이용
            text += page.extract_text()
    return text

def get_text_chunk(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # embedding API 선택
    # 선택 1: OpenAI embedding API 사용시 (유료)
    embeddings = OpenAIEmbeddings()

    # 선택 2: 허깅페이스에서 제공하는 Instructor embedding API 사용시 (무료)
    # 성능은 OpenAIEmbeddings보다 우수하지만 느리다.
    # https://huggingface.co/hkunlp/instructor-xl
    # model_name 인자값으로 hkunlp/instructor-xl을 입력
    # 단, 2개의 dependency를 설치해야 함 pip install instructorembedding sentence_transformers
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# get_conversation_chain(_vectorstore) 실행에 필요한 모듈
# (deprecated) from langchain.chat_models import ChatOpenAI # ConversationBufferMemory()의 인자로 들어갈 llm으로 ChatOpenAI모델을 사용하기로 함
from langchain_community.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory # 대화내용을 저장하는 memory
from langchain.chains import ConversationalRetrievalChain

######################################################
#  핵심 함수 GCC : get_conversation_chain(_vectorstore) #
######################################################
# chain 생성 3요소(LLM, retriever, memory) --> ConversationalRetrievalChain.from_llm(chain 생성 3요소)
# 출력 : 질의응답을 담당하는 ConversationalRetrievalChain.from_llm(3요소) --> conversation_chain 객체 반환
def get_conversation_chain(_vectorstore):
   # 메모리에 로드된 env파일에서 "OPENAI_API_KEY"라고 명명된 값을 변수로 저장
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # 선택 1: 대화에 사용될 llm API 객체를 llm 변수에 저장
    llm = ChatOpenAI(
        temperature=0.1,    # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4-turbo-preview", # chatGPT-4 Turbo 사용
        openai_api_key=OPENAI_API_KEY # Automatically inferred from env var OPENAI_API_KEY if not provided.
        )

    # 선택 2: HuggingFaceHub를 llm 모델로 사용시
    # from langchain.llms import HuggingFaceHub
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    # ConverstaionBufferMemory 클래스를 이용하여 대화내용을 chat_history라는 key값으로 저장해주는 memory 객체를 생성
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # llm 객체, memory 객체를 인자로 입력하여 DB 검색결과를 출력하는 ConversationalRetrievalChain.from_llm 객체를 생성
    conversation_chain = ConversationalRetrievalChain.from_llm(
        # 검색을 실행하는 llm 선택
        llm=llm,
        # 검색을 당하는 vector DB를 retriver 포맷으로 저장
        retriever=_vectorstore.as_retriever(),
        # 사용자와 대화내용을 메모리에 저장하여 같은 맥락에서 대화를 유지
        memory=memory
    )
    # ConversationalRetrievalChain.from_llm 객체로 생성된 convestaion_chain을 반환
    return conversation_chain

######################################################
#              핵심 함수 conversation_window              #
######################################################
# GCC함수를 통해 생성된 user와의 대화를 st.session_state에 별도로 저장 --> 프론트에 뿌려줌

def conversation_window(user_question) :
    # 질의응답 역할: ConversationalRetrievalChain.from_llm() 객체 생성
    # --> conversation_chain 객체
    # --> st.sessioin_state.conversation에 저장
    # 질문 --> ({'question': user_question}) 형태로 인자에 넣어주면 결과를 출력하고, 대화내용은 메모리에 저장
    # 질문+답변 --> response['chat_history']에는 저장

    ##################
    #    질문 저장     #
    ##################
    # main() 함수 맨마지막에 st.session_state.conversation = get_conversation_chain(vectorstore) 에 의하여
    # st.session_state.conversation에는 '질의응답'이 아니라 conversation_chain '함수' 그 자체가 저장되어 있음
    # 따라서 conversation_chain() = ConversationalRetrievalChain.from_llm() 이므로
    # conversation_chain()에서 ()안에 {'question': user_question} 형태로 인자를 넣어서 질문
    response = st.session_state.conversation({'question': user_question})

    # ConversationalRetrievalChain.from_llm()는 응답을 "객체로 반환"함.
    # 따라서 response 안에는 응답객체가 저장되어 있으며, "객체의 key값이 chat_history"에 대응되는 value로서 질의/응답이 저장됨.
    # 확인 --> st.write(response) 해보면 chat_history라는 key값에 질의/응답이 저장되어 있음을 알수있다.

    ##################
    #                #
    ##################
    # 응답객체에서 'chat_history'만을 추출한 후, st.session_state에서 별도로 누적적으로 보관하여 전체 대화를 기록함
    st.session_state.chat_history = response['chat_history']

    # message 객체의 content 속성에 대화가 들어있으므로 이를 추출하여 탬플릿의 {{MSG}} 위치에 넣는 replace 메소드를 사용하여 대체
    for i, message in enumerate(st.session_state.chat_history):

        # 0부터 시작하므로 사용자 질의는 항상 짝수번째 기록
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

        # bot의 응답은 항상 홀수번째 기록
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

######################################################
#                    admin key 검증                   #
######################################################

def is_admin(_input_key):
    # 문자열을 byte열로 encoding을 먼저 실시한 후, sha256으로 암호화
    input_key_hash = hashlib.sha256(_input_key.encode()).hexdigest()
    saved_key_hash = hashlib.sha256(os.getenv("OPENAI_API_KEY").encode()).hexdigest()
    if input_key_hash == saved_key_hash :
        return True
    else :
        return False

###############################
#          (참고)  채팅창        #
###############################
# 사전에 정의한 css, html양식을 st.wirte() 함수의 인자로 넣어주면 웹사이트 형식으로 출력한다.
# st.write(user_template.replace("{{MSG}}", "Hellow Bot"), unsafe_allow_html=True)
# st.write(bot_template.replace("{{MSG}}", "Hellow Human"), unsafe_allow_html=True)

######################################################
#                        Main                        #
######################################################

def main() :
    load_dotenv()
    st.set_page_config(page_title="TONchat", page_icon=":books:", layout="wide")
    # css, html관련 설정은 실제 대화관련 함수보다 앞에서 미리 실행해야 한다.
    st.write(css, unsafe_allow_html=True)


    #####################################################
    #               st.session_state 초기화               #
    #####################################################

    # main() 함수 맨 아래에 있는 st.session_state.conversation = get_conversation_chain(vectorstore)을 통해
    # session_state 객체의 속성으로 conversation 속성이 생성. 그 안에 딕셔너리로 질의/응답이 저장된다.
    # { question : ddd, answer : ddd } 이런식이다.

    # 실행에 앞서 conversation 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # 실행에 앞서 chat_history 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    ###############################
    #            질문창             #
    ###############################
    # 질문입력창
    st.header("TONchat")
    st.write("Ask a question about Tokamak Network's services")
    st.write("- Titan L2 Network")
    st.write("")

    # st.text_input()에 질문이 입력되면 True를 반환
    user_question = st.text_input("Input your question")

    # 질문이 들어오면 if문이 true가 되고, 질문에 대한 답변을 처리한다.
    if user_question:
        conversation_window(user_question)


    ###############################
    #      sidebar 파일 업로드       #
    ###############################
    with st.sidebar:
        with st.popover("Admin login"):
            st.markdown("Admin key 🔑")

            # 세션 상태에 admin 값이 없으면 초기화
            if 'admin' not in st.session_state:
                st.session_state.admin = False

            # 입력 필드 값 변경 감지
            admin_key = st.text_input("Input your admin key")
            if st.button("Login"):
                st.session_state.admin = is_admin(admin_key)

        if st.session_state.admin:
            st.write("Hi, Admin !")
            if st.button("Logout", type='primary'):
                st.session_state.admin = False
                # 로그아웃 후 즉시 스크립트 재실행
                st.experimental_rerun()

            st.header("Your documents")
            # upload multiple documents
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'process'", accept_multiple_files=True)
            if st.button("Process") :
                with st.spinner('Processing') :

                    # DB 생성 3단계
                    ##########################
                    #     1. get pdf text    #
                    ##########################
                    raw_text = get_pdf_text(pdf_docs)

                    ###########################
                    #  2. get the text chunks #
                    ###########################
                    text_chunks = get_text_chunk(raw_text)
                    st.write(text_chunks)

                    ###########################
                    #  3. create vector store #
                    ###########################
                    vectorstore = get_vectorstore(text_chunks)

                    #############################################################################
                    #  conversation chain 으로 대화 --> st.session_state에 기록 --> 프론트에 대화 출력  #
                    #############################################################################
                    # 핵심함수 get_conversation_chain() 함수를 사용하여, 첫째, 이전 대화내용을 읽어들이고, 둘째, 다음 대화 내용을 반환할 수 있는 객체를 생성
                    # 다만 streamlit 환경에서는 input이 추가되거나, 사용자가 버튼을 누르거나 하는 등 새로운 이벤트가 생기면 코드 전체를 다시 읽어들임
                    # 이 과정에서 변수가 전부 초기화됨.
                    # 따라서 이러한 초기화 및 생성이 반복되면 안되고 하나의 대화 세션으로 고정해주는 st.sessiion_state 객체안에 대화를 저장해야 날아가지 않음
                    # conversation이라는 속성을 신설하고 그 안에 대화내용을 key, value 쌍으로 저장 (딕셔너리 자료형)
                    st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()