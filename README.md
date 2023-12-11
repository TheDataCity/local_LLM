# Company Chatbot - Local LLM 🌐

## Introduction
This is a chatbot application that provides information about companies based on their website content. Users can select a company and ask questions, and the chatbot will extract and analyze information from the company's website to provide detailed responses. 

**BEST PART - It does not use Open AI API!!!!!!!!!! Isn't the cool, and you can get JSON output tooooo!!!!!**

## Features
- **Total Data Security**: Your information stays solely on your device, guaranteeing absolute security.
- **Flexible Model Compatibility**: Effortlessly incorporate various open-source models like HF, GPTQ, GGML, and GGUF.
- **Wide Range of Embeddings**: Access a selection of open-source embeddings.
- **LLM Reusability**: Once downloaded, your Large Language Model can be reused without needing to redownload.
- **Session-Based Conversation Memory**: Keeps track of your past discussions within a session.
- **API Functionality**: LocalGPT offers an API for developing RAG Applications.
- **User-Friendly Interfaces**: LocalGPT includes two graphical user interfaces, one utilizing the API and another standalone version built with streamlit.
- **Broad Hardware Compatibility**: Ready-to-use on various platforms, enabling you to engage with your data using `CUDA`, `CPU`, `MPS`, and more.

The code can run on command line as well as GUI (streamlit applicatiobn)

## Getting Started
- Clone the repository to your local machine.
```shell
git clone https://github.com/TheDataCity/local_LLM.git
```
- Set up the virtual environment and activate it
```shell
python<version> -m venv <virtual-environment-name>
```
- Install the dependencies using pip
```shell
pip install -r requirements.txt
```
- Install LLAMA-CPP
```shell
pip install llama-cpp-python
```

## Creating the Vector Database
The database folder will contain text files extracted from various companies' websites. These text files will be used by the chatbot to provide information about specific company.

**Location**: The database folder is expected to be located in the root directory of this project. Sample of two companies is provided in the repository and move it accordingly to the database folder. The project folder structure should look like this:<br/>
project-root/<br/>
│<br/>
├── data/<br/>
│ └── text_file_database<br/>
│ │   └── company_1_info.txt<br/>
│ │   └── company_2_info.txt<br/>
│ │   └── ...<br/>

**Run the program**: In the virtual environment run the "createVectorDb.py" file
```shell
python createVectorDb.py
```
**Note**
Use the device type argument to specify a given device.
To run on `cpu`
```shell
python createVectorDb.py --device_type cpu
```

To run on `M1/M2`
```shell
python createVectorDb.py --device_type mps
```

Use help for a full list of supported devices.
```sh
python createVectorDb.py --help
```

Once the code is executed sucessfully you will find the "vector_database" folder created within the "data" folder.

## Executing the code for Command line interface.
In order to chat using command line, run the following command (by default, it will run on `cuda`).
```shell
python run_localLLM.py
```
You can also specify the device type just like `ingest.py`
```shell
python run_localGPT.py --device_type mps # to run on Apple silicon
```
