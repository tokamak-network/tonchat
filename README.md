# 01. Introduction

TONchat is a question and answer service based on the chatGPT API. It not only stores code information such as various smart contracts developed by Tokamak Network and related annotations, but also various community-centered documents such as user guides and Q&As in a database and utilizes them as response resources.

This configuration is commonly referred to as a RAG (Retrieval Augmented Generation) application and is one of the application services that utilizes LLM (Large Language Model) based AI as it is rapidly growing in recent years.

In other words, a typical model such as chatGPT has two limitations: first, it does not contain any additional information after the cut off date when training ends, and second, it cannot answer questions about specialized or fringe areas beyond the information utilized for training.

Therefore, RAG will build a separate DB related to such specific expertise, and the LLM engine will search for it first, and additional knowledge will be supplemented by previously learned knowledge or internet searches. This is an advantageous way to respond to specialized knowledge in specific fields.

TONchat is expected to open the alpha and beta versions in the first quarter of 2024, followed by updates in the CI/CD field, such as the automatic collection of learning data, and feedback from users, and to complete development in the fourth quarter of 2024.

After completion of development, we plan to transfer all rights to the development and use of TONchat (including commercial utilization) to the Tokamak Network's community. This will allow various knowledge and know-how generated within the community to be accumulated and easily shared, thereby removing barriers to not only the convenience of existing users but also the influx of new users.

TONchat has only one goal. To make Tokamak Network's source code more accessible to the community.

---------------------------------------------------------
TONchat은 chatGPT API를 기반으로 운영되는 질의응답 서비스입니다. Tokamak Network가 개발한 다양한 스마트컨트랙트와 관련 주석처리와 같은 code 정보뿐만 아니라, user guide 및 질의응답 등 커뮤니티 중심으로 형성된 다양한 활용문서를 database로 저장하여 응답자료로 활용합니다.

이와 같은 구성은 흔히 RAG(Retrieval Augmented Generation) application으로 불리고 있으며, 최근 LLM(Large Language Model) 기반의 AI가 급성장하면서, 이를 활용하는 응용서비스 중의 하나입니다.

즉, chatGPT와 같인 일반적인 모델은 첫째, 학습이 종료된 그 날짜(cut off date)이후로는 추가적인 정보를 담고 있지 않다는 단점, 둘째, 학습에 활용된 정보이외의 전문적이거나 지엽적인 특정분야에 대한 질문에 답할 수 없다는 한계가 있습니다.

따라서 RAG에서는 이러한 특정 전문지식에 관련된 DB를 별도로 구축하여 LLM engine이 1차적으로 검색하고, 추가적인 지식은 기존에 학습한 지식 혹은 인터넷 검색을 통해 보완하는 형태로 진행하게 됩니다. 따라서 특정분야의 전문지식에 대한 응답에 있어서 유리한 방식입니다.

TONchat은 2024년 1분기에 알파버전, 베타버전을 차례로 오픈하고, 이후에는 학습데이터의 자동수집 기능 등 CI/CD 분야의 업데이트, 사용자의 피드백을 수렴하여 2024년 4분기에 개발을 완료할 예정입니다.

개발완료 이후에는 Tokamak Network의 커뮤니티로 TONchat의 개발 및 이용에 대한 모든 권한(상업적 활용을 포함)을 이관할 계획입니다. 이를 통해 커뮤니티 내에서 생성된 다양한 지식과 노하우가 축적되고 손쉽게 공유되도록 하여, 기존유저의 편의뿐만 아니라 신규유저의 유입에 있어서도 장애물을 없애려고 합니다.

TONchat의 목표는 단 한가지 입니다. Tokamak Network의 source code를 커뮤니티가 좀더 쉽게 활용할 수 있도록 지원하는 것입니다.

# 02. Automation
## Configuration
TBD

## How to manage
TBD
# 03. Governance
TBD

# 04. Installation
### 01. Activate virtual environment
It is recommended to use Anaconda virtual environment with python version 3.10

    conda create -n {{your-virtual-environment}} python=3.10

    conda activate {{your-virtual-environment}}

### 02. Installation of packages

At first, check your pip location is under your virtual environment as below :

    ~/opt/anaconda3/envs/{{your-virtual-environment}}/bin/pip

With the virtual enviroment, you can use its bundle pip and install packages using it as below.


    pip install -r requirements.txt

### 03. Check installation

Packages installed using pip are located in the virtual environment directory. You can check the installation as below :

    pip list -v

~/opt/anaconda3/envs/{{your-virtual-environment}}/lib/python3.12/site-packages

Other packages installed using conda are located in the site-packages directory as below :

    conda list

~/opt/anaconda3/envs/{{your-virtual-environment}}


