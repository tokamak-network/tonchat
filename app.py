import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings # embedding에 사용할 2개의 API
# 페이스북이 만든 고속 Vector DB인 faiss를 랭체인에서 래핑한 라이브러리(구별을 위해 대문자 표기)
# 래핑함수이므로 원래함수인 faiss-cpu가 이미 설치되어 있어야 함 pip install faiss-cpu
from langchain.vectorstores import FAISS # importing FAISS from langchain root module is no longer supported
from htmlTemplates import css, bot_template, user_template   # htmlTemplates.py 파일안에 있는 모듈들을 가져옴
import os # host server안에 db 폴더 생성하는 st.file_uploader.getbuffer() 사용시 필요
import numpy as np

# 전역변수 : FAISS 인덱스 파일 경로
VECTORDB_PATH = './vectordb'

##############################################################
#                      openAPI Key 입력                       #
##############################################################
# 첫째, OPENAI_API_KEY = "....." 형태로 값을 지정하여 .env 파일로 저장
# 둘째, os.getenv 함수에 .env파일이 위치한 PATH 값을 입력
OPENAI_API_KEY_PATH = "../tonchat_key/.env"
# 셋째, .env 파일을 현재 실행환경에 등록. 그 결과, 개별 API 함수의 인자에 KEY값을 일일이 넣지 않아도 됨
load_dotenv(OPENAI_API_KEY_PATH)
# 넷째, 현재 실행환경에서 "OPENAI_API_KEY"라고 명명된 값을 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# 만약 streamlit 온라인 호스팅시에는 openAI API key 입력이 별도로 있으므로 위와같은 세팅이 불필요

####################################################################################
# STEP 1 : st.file_uploader에 의해 생성된 pdf_docs 객체에 pdf원본파일을 기록하여 서버에 저장
def save_pdf(_pdf_docs):
    # 업로드한 pdf 원본을 저장하는 data 폴더가 존재하지 않으면 생성
    if not os.path.exists('data'):
        os.makedirs('data')

    # uploadedfile.name 속성 이용하여 업로드된 파일을 하나씩 지정하여 binary 파일을 생성한 후
    # wb 옵션으로 binary writting이 가능하도록 한 상태로 일단 open
    # uploaded_file.getbuffer()을 이용하여 업로드한 파일의 내용을 바이트 형태로 반환받고
    # 이를 f.write에서는 binary로 읽어와서 data' 디렉토리에 있는 새 파일에 복사함
    # 결과적으로 data 폴더안에 해당이름의 pdf파일이 복사되어 생성됨
    for uploaded_file in _pdf_docs:
        with open(os.path.join('data', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            # 제대로 저장되었는지 확인하기 위하여 저장완료된 파일의 앞부분 글자 1000개까지 출력
            # pdf 파일은 text가 아니므로 encoding옵션이 필요없는 대신, binary이므로 read 옵션은 rb로 설정
            # with open(os.path.join('data', uploaded_file.name), 'rb') as file:
            #     content = file.read()
            #     st.write(content[:1000])

####################################################################################
# STEP 2 : 업로드된 여러개의 pdf파일 리스트인 pdf_docs객체 --> 이에 대해 text만 추출하는 과정
def get_pdf_text(_pdf_docs):
    # 빈 text 배열 생성
    text = ""
    for pdf in _pdf_docs: # 각 pdf 파일
        # 각 pdf 파일을 일단 PdfReader클래스의 객체로 생성
        pdf_reader = PdfReader(pdf)
        # PdfReader 객체의 pages() 메소드 활용 --> 각 페이지별 내용을 추출
        for page in pdf_reader.pages:
            # 각 페이지 객체의 extract_text() 메소드 활용 --> 텍스트를 추출하여 모든 text라는 하나의 파일에 저장 (이때, 페이지별로 구분되어 저장됨)
            text += page.extract_text()
    # 테스트 : 추출된 text의 앞부분 200글자 출력
    # st.write('get_pdf_text : ', text[:200])
    return text

####################################################################################
# STEP 3 : 추출된 text --> CTS 객체 생성 --> split_text() 메소드 이용 --> chunk단위로 반환
def get_text_chunk(_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(_text)
    # 테스트 출력 : 추출된 chunk 중에서 앞부분 10개 출력
    # st.write(chunks[:10])
    return chunks

####################################################################################
# STEP 4 : chunk를 vector DB로 변환 및 저장
# FAISS 방식으로 vectorized된 DB(FAISS Index라고 호칭)
def text_to_vectorstore(_text_chunks):
    # 선택 1: OpenAI embedding API 사용시 (유료)
    embeddings = OpenAIEmbeddings()
    # 선택 2: 허깅페이스에서 제공하는 Instructor embedding API 사용시 (무료)
    # 성능은 OpenAIEmbeddings보다 우수하지만 느리다.
    # https://huggingface.co/hkunlp/instructor-xl
    # model_name 인자값으로 hkunlp/instructor-xl을 입력
    # 단, 2개의 dependency를 설치해야 함 pip install instructorembedding sentence_transformers
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=_text_chunks, embedding=embeddings)




    # 테스트 : plain text로 변환된 상태인지 binary상태인지 파일타입을 확인
    st.write(type(vectorstore))




    return vectorstore

# vectorDB 저장 : vector DB 파일명을 arguement로 입력
def save_vectorstore(_vectorstore):
    _vectorstore.save_local(VECTORDB_PATH)

####################################################################################
# STEP 5 : vector DB 로딩 및 업데이트
def load_vectorstore():
    # vector DB가 존재하지 않는 경우, 최초로 vectorstore를 초기화하여 저장 (공백문자 초기화 불가)
    if not os.path.exists(os.path.join(VECTORDB_PATH, "index.faiss")):
        vectorstore_init = text_to_vectorstore("initialized") # initialized 문자로 초기화하여 저장 (공백문자로 초기화 불가)
        save_vectorstore(vectorstore_init)
        loaded_vectorstore = vectorstore_init
    else:
        embeddings = OpenAIEmbeddings()
        #load_local()은 index.faiss가 저장된 폴더명을 인자로 받는다
        loaded_vectorstore = FAISS.load_local(VECTORDB_PATH, embeddings=embeddings)
    return loaded_vectorstore

def update_vectorstore(_new_index):
    # 로컬에 저장된 vectordb가 있는 경우, 신규 db와 merge하여 save
    if os.path.exists(os.path.join(VECTORDB_PATH, "index.faiss")):
        embeddings = OpenAIEmbeddings()

        # load 할 때, embeddings를 이용해서 binary를 plain text로 변환해서 불러온다.
        loaded_vectorstore = FAISS.load_local(VECTORDB_PATH, embeddings=embeddings)

        # 테스트 : 실제로 출력해보면 binary가 아니라 plain text가 출력됨을 알 수 있다.
        st.write("def update_vectorstore's loaded_vectorstore variable is :", loaded_vectorstore)

        # 신규 data인 _vectorstore는 bianry 파일 --> ._dict() 메소드를 활용하여 확인필요
        # 로딩된 loaded_vectorstore객체는 plain text 파일
        # 이 상태에서 merge 하면 에러발생 --> ValueError: Cannot merge with this type of docstore
        # 신규 data인 _vectorstore는 bianry 파일 --> plain text로 변환하여 merge해야 함




        # _vectorstore는 get_vectorestore()함수를 통해 생성된 객체이므로, 이미 plain text로 변환된 상태인지 확인 필요
        st.write('vectorstore type is :', type(_new_index))
        st.write('loaded_vectorstore type is :', type(loaded_vectorstore))

        ################################################################
        # 기존 db인 index.faiss 불러오기
        ################################################################

        # _vectorstore는 plain text로 변환된 상태이므로, merge_from 메소드를 사용하여 merge
        # faiss 파일을 binary로 "read" 하는 것과, text파일로 "load" 하는 것은 다르다.
        # merge_from은 binary 상태의 index.faiss 파일에서만 가능하다. 따라서 저장된 faiss.index 파일을 "read"로 가져와야 한다.
        import faiss
        old_index = faiss.read_index(VECTORDB_PATH)

        ################################################################
        # 신규 db인 new.faiss와 merge 하기
        ################################################################

        updated_index = old_index.merge_from(_new_index)




        # updated_vectorstore가 None type이라서 save_vectore()실행이 안되는 에러 발생
        # 따라서 updated_vectorstore가 혹시 None이 아닌지 체크
        if updated_vectorstore is not None:
            save_vectorstore(updated_vectorstore)
        else:
            print("updated_vectorstore is None")

    # 로컬에 저장된 vectordb가 없는 경우, vectorstore를 최초로 save
    else :
        save_vectorstore(_vectorstore)
        updated_vectorstore = _vectorstore
    # 업데이트된 faiss index 파일을 리턴
    return updated_index


#####################
#  핵심 함수 GCC_pdf  #
#####################
# 입력 : VectorDB
# 출력 : 질문을 입력하면 입력된 상기 VectorDB에 대한 검색과 결과출력을 담당하는 객체를
# CRC(ConversationalRetrievalChain) 객체의 .from_llm의 메소드를 이용하여 생성하여 반환함

from langchain.chat_models import ChatOpenAI # ConversationBufferMemory()의 인자로 들어갈 llm으로 ChatOpenAI모델을 사용하기로 함
from langchain.memory import ConversationBufferMemory # 대화내용을 저장하는 memory
from langchain.chains import ConversationalRetrievalChain # 내부 DB를 참조하여 chatGPT대화를 진행
from langchain.chains import ConversationChain # 내부 DB없이 일반적인 chatGPT대화를 진행

def get_conversation_chain_pdf(_loaded_vectorscore):

    # 선택 1: 대화에 사용될 llm API 객체를 llm 변수에 저장
    llm = ChatOpenAI(model_name='gpt-4')
    # 선택 2: HuggingFaceHub를 llm 모델로 사용시
    # from langchain.llms import HuggingFaceHub
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    # 대화내용 저장 memory 객체 생성 : ConverstaionBufferMemory 클래스를 이용, 대화를 chat_history라는 key값으로 저장
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # llm 객체, memory 객체를 인자로 입력하여 DB 검색결과를 출력하는 ConversationalRetrievalChain.from_llm 객체를 생성
    conversation_chain_pdf = ConversationalRetrievalChain.from_llm(
        # 검색도구인 llm 모델의 종류를 입력
        llm=llm,
        # 검색대상인 vector DB를 retriver 포맷으로 변환
        retriever = _loaded_vectorscore.as_retriever(),
        # 사용자와 대화내용을 메모리에 저장하여 같은 맥락에서 대화를 유지
        memory=memory
    )
    # ConversationalRetrievalChain.from_llm 객체인 convestaion_chain을 반환
    return conversation_chain_pdf


###########################
#       보조함수 GCC        #
###########################

    # (참고)  채팅창
    # 사전에 정의한 css, html양식을 st.wirte() 함수의 인자로 넣어주면 웹사이트 형식으로 출력한다.
    # st.write(user_template.replace("{{MSG}}", "Hellow Bot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hellow Human"), unsafe_allow_html=True)

def get_conversation_chain():

    # 선택 1: 대화에 사용될 llm API 객체를 llm 변수에 저장
    llm = ChatOpenAI(model_name='gpt-4')
    # 선택 2: HuggingFaceHub를 llm 모델로 사용시
    # from langchain.llms import HuggingFaceHub
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    # 대화내용 저장 memory 객체 생성 : ConverstaionBufferMemory 클래스를 이용, 대화를 history라는 key값으로 저장
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    # llm 객체, memory 객체를 인자로 입력하여, 내부 DB없이 대화를 연속적인 대화를 생성하는 객체를 반환
    conversation_chain = ConversationChain(llm=llm, memory=memory)

    return conversation_chain


##########################################
# 핵심함수 handle_userinput : 1. 기본 응답    #
##########################################

def handle_userinput(_user_question) :
    # 질문을 입력하면 DB 검색과 결과출력을 담당하는 ConversationalRetrievalChain.from_llm의 객체 -> st.sessioin_state.conversation
    # 이 객체의 메소드로서 사용자의 질문은 ({'input': user_question}) 형태로 인자에 넣어주면 결과를 출력하고, 대화내용은 메모리에 저장된다.
    # response['chat_history']에는 사용자의 질문과 대답이 저장되어 있다.
    response = st.session_state.conversation({'input': _user_question}) # st.settion_state객체의 내장메소드에 사용자 질문을 받는 기능이 있을 것임
    # st.write(response) # 딕셔너리로 출력되며 chat_history라는 key값에 질의/응답이 저장되어 있음을 알수있다.

    # 'chat_history'를 key값으로 하여 이번의 질의응답만 저장되어 있는데, 이를 메모리에 누적에서 보관하여 전체 대화를 기록함
    st.session_state.history = response['history']

    # message 객체의 content 속성에 대화가 들어있으므로 이를 추출하여 탬플릿의 {{MSG}} 위치에 넣는 replace 메소드를 사용
    for i, message in enumerate(st.session_state.history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


##################################################
# 핵심함수 handle_userinput : 2. 로컬 pdf 활용한 응답  #
##################################################
# 인자인 user_question는 user로부터 받은 질문이며, 이를 인자로 넣으면 대답을 반환하는 함수
# 내부에 GCC함수가 들어 있고, 대화 세션을 유지하는 st.session_sate.converstaion 메소드를 통해
# 즉, 겉으로는 안드러나지만 st.session_sate.converstaion안에는 GCC 함수가 있음

def handle_userinput_pdf(_user_question) :
    # 질문을 입력하면 DB 검색과 결과출력을 담당하는 ConversationalRetrievalChain.from_llm의 객체 -> st.sessioin_state.conversation
    # 이 객체의 메소드로서 사용자의 질문은 ({'question': user_question}) 형태로 인자에 넣어주면 결과를 출력하고, 대화내용은 메모리에 저장된다.
    # response['chat_history']에는 사용자의 질문과 대답이 저장되어 있다.
    response = st.session_state.conversation({'question': _user_question}) # st.settion_state객체의 내장메소드에 사용자 질문을 받는 기능이 있을 것임
    # st.write(response) # 딕셔너리로 출력되며 chat_history라는 key값에 질의/응답이 저장되어 있음을 알수있다.

    # 'chat_history'를 key값으로 하여 이번의 질의응답만 저장되어 있는데, 이를 메모리에 누적에서 보관하여 전체 대화를 기록함
    st.session_state.chat_history = response['chat_history']

    # message 객체의 content 속성에 대화가 들어있으므로 이를 추출하여 탬플릿의 {{MSG}} 위치에 넣는 replace 메소드를 사용
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


#######################################################################################
#######################################################################################
#                                        Main                                         #
#######################################################################################
#######################################################################################


def main() :
    load_dotenv()
    st.set_page_config(page_title="TONchat", page_icon=":books:", layout="wide")
    # css, html관련 설정은 실제 대화관련 함수보다 앞에서 미리 실행해야 한다.
    st.write(css, unsafe_allow_html=True)

    ###############################
    #           0. 초기화           #
    ###############################

    # st.session_state.conversation = get_conversation_chain(vectorstore)을 통해
    # sesstion_state 객체의 속성으로 conversation이 신설되고, 그 안에 딕셔너리로 질의/응답이 저장된다.
    # { question : ddd, answer : ddd } 이런식이다.
    # 이러한 저장이 이뤄지도록 일단 conversation 속성에 None으로 초기화를 시켜 준비해놓는다.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    if 'history' not in st.session_state:
        st.session_state.history = None


    ###############################
    #         1. Question         #
    ###############################
    # 질문입력창
    st.header("TONchat")
    user_question = st.text_input("Ask a question about your documents")


    ##############################
    #         2. Answer          #
    ##############################
    loaded_vectorscore = load_vectorstore()
    st.session_state.conversation = get_conversation_chain_pdf(loaded_vectorscore)
    # 질문이 입력되어있다면 if문이 true가 되고, 질문에 대한 답변을 처리한다.
    if user_question:
        handle_userinput_pdf(user_question)
    # 로컬에 저장된 vectordb가 없는 경우 -> GCC()함수를 사용하여 사용자질문에 답변한다
    else :
        st.session_state.conversation = get_conversation_chain()
        # 질문이 입력되어있다면 if문이 true가 되고, 질문에 대한 답변을 처리한다.
        if user_question:
            handle_userinput(user_question)


    ####################################################
    #  3. uploading pdf + updating vectorDB --> Q & A  #
    ####################################################
    with st.sidebar:
        st.header("Your documents")

        ###############################
        #    Upload pdf to memory     #
        ###############################
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'process'", accept_multiple_files=True)

        ###############################
        #          Process pdf        #
        ###############################
        if st.button("Process") :
            with st.spinner('Processing') :

                ##############################
                # 01.save PDFs to the folder #
                ##############################

                save_pdf(pdf_docs)

                ############################
                #     02. get pdf text     #
                ############################
                raw_text = get_pdf_text(pdf_docs)

                ############################
                #  03.Split PDFs to chunks #
                ############################
                text_chunks = get_text_chunk(raw_text)
                st.write(text_chunks)

                ############################
                #  04. create vectorstore  #
                ############################
                new_vectorstore = text_to_vectorstore(text_chunks)

                ############################
                #  05. update vectorstore  #
                ############################
                update_vectorstore(new_vectorstore)

                ############################
                #    06. Conversation      #
                ############################
                # 핵심함수 get_conversation_chain_pdf() 함수를 사용하여, 첫째, 이전 대화내용을 읽어들이고, 둘째, 다음 대화 내용을 반환할 수 있는 객체를 생성
                # 다만 streamlit 환경에서는 input이 추가되거나, 사용자가 버튼을 누르거나 하는 등 새로운 이벤트가 생기면 코드 전체를 다시 읽어들임
                # 이 과정에서 변수가 전부 초기화됨.
                # 이를 방지하고자 하나의 대화 세션으로 고정해주는 st.settion_state 객체안에 conversation이라는 속성을 신설하고
                # 그 안에 대화내용을 key, value 쌍으로 저장 (딕셔너리 자료형)
                loaded_vectorstore = load_vectorstore()
                st.session_state.conversation = get_conversation_chain_pdf(loaded_vectorstore)


# app.py가 모듈로서 간접활용되지 않고, 운영체제에서 프로그램으로서 직접 실행되는 경우에만 main() 함수를 실행
if __name__ == "__main__":

    main()