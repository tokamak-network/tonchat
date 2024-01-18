# Introduction

TONchat은 chatGPT API를 기반으로 운영되는 질의응답 서비스입니다. Tokamak Network가 개발한 다양한 스마트컨트랙트와 관련 주석처리와 같은 code 정보뿐만 아니라, user guide 및 질의응답 등 커뮤니티 중심으로 형성된 다양한 활용문서를 database로 저장하여 응답자료로 활용합니다.

이와 같은 구성은 흔히 RAG(Retrieval Augmented Generation) application으로 불리고 있으며, 최근 LLM(Large Language Model) 기반의 AI가 급성장하면서, 이를 활용하는 응용서비스 중의 하나입니다.

즉, chatGPT와 같인 일반적인 모델은 첫째, 학습이 종료된 그 날짜(cut off date)이후로는 추가적인 정보를 담고 있지 않다는 단점, 둘째, 학습에 활용된 정보이외의 전문적이거나 지엽적인 특정분야에 대한 질문에 답할 수 없다는 한계가 있습니다.

따라서 RAG에서는 이러한 특정 전문지식에 관련된 DB를 별도로 구축하여 LLM engine이 1차적으로 검색하고, 추가적인 지식은 기존에 학습한 지식 혹은 인터넷 검색을 통해 보완하는 형태로 진행하게 됩니다. 따라서 특정분야의 전문지식에 대한 응답에 있어서 유리한 방식입니다.

TONchat은 2024년 1분기에 알파버전, 베타버전을 차례로 오픈하고, 이후에는 학습데이터의 자동수집 기능 등 CI/CD 분야의 업데이트, 사용자의 피드백을 수렴하여 2024년 4분기에 개발을 완료할 예정입니다.

개발 이후에는 Tokamak Network의 커뮤니티로 TONchat의 개발 및 이용에 대한 모든 권한(상업적 활용을 포함)을 이관할 계획입니다. 이를 통해 커뮤니티 내에서 생성된 다양한 지식과 노하우가 축적되고 손쉽게 공유되도록 하여, 기존유저의 편의뿐만 아니라 신규유저의 유입에 있어서도 장애물을 없애려고 합니다.

TONchat의 목표는 단 한가지 입니다. Tokamak Network의 source code를 커뮤니티가 좀더 쉽게 활용할 수 있도록 지원하는 것입니다.

# Installation

pip install -r requirements.txt

