    ###############################
    #      sidebar 파일 업로드       #
    ###############################
    with st.sidebar:
        # Admin login
        with st.popover("Admin login"):
            st.markdown("Admin key 🔑")
            admin = is_admin(st.text_input("Input your admin key"))
            # login_count Initialization
            if 'login_count' not in st.session_state:
                st.session_state.login_count = 0

        if admin:
            st.session_state.login_count += 1
            if st.session_state.login_count % 2 == 1:
                st.write("Hi, Admin !")
                # 버튼이 클릭되면 카운터를 증가
                if st.button("Logout", type="primary"):
                    st.session_state.login_count += 1

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