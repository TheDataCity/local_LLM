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

# Getting Started
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

# Creating the Vector Database
The database folder will contain text files extracted from various companies' websites. These text files will be used by the chatbot to provide information about specific company.

**Location**: The database folder is expected to be located in the root directory of this project. Sample of two companies is provided in the repository and move it accordingly to the database folder. The project folder structure should look like this:<br/>
project-root/<br/>
│<br/>
├── data/<br/>
│ └── text_file_database/<br/>
│ │   └── company_1_info.txt<br/>
│ │   └── company_2_info.txt<br/>
│ │   └── ...<br/>

**Run the program**: In the virtual environment run the "createVectorDb.py" file
```shell
python createVectorDb.py
```
**Note:**
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

Here's a brief explanation for each device type:
- **CPU**: Standard processing unit used in most computers.
- **CUDA**: NVIDIA's parallel computing platform for general computing on GPUs (Graphics Processing Units).
- **IPU**: Intelligence Processing Unit, designed for machine learning and AI tasks.
- **XPU**: A general term for any type of processing unit including CPUs, GPUs, and specialized accelerators.
- **MKLDNN**: Math Kernel Library for Deep Neural Networks, optimized for high performance on Intel CPUs.
- **OpenGL**: A cross-language, cross-platform API for rendering 2D and 3D vector graphics.
- **OpenCL**: Open Computing Language, used for programming tasks across different platforms like CPUs and GPUs.
- **IDeep**: Intel Deep Learning Inference Engine, for optimized deep learning inference on Intel hardware.
- **HIP**: Heterogeneous-compute Interface for Portability, a runtime API for porting CUDA applications to AMD GPUs.
- **VE**: Vector Engine, typically used in high-performance computing.
- **FPGA**: Field-Programmable Gate Array, an integrated circuit designed to be configured after manufacturing.
- **ORT**: ONNX Runtime, a cross-platform, high-performance scoring engine for Open Neural Network Exchange models.
- **XLA**: Accelerated Linear Algebra, a domain-specific compiler for linear algebra that optimizes TensorFlow computations.
- **Lazy**: Refers to lazy evaluation in computing, where computation is delayed until necessary.
- **Vulkan**: A low-overhead, cross-platform API for high-performance 3D graphics and compute tasks.
- **MPS**: Metal Performance Shaders, used for efficient graphic and compute tasks on Apple GPUs.
- **Meta**: A device type in PyTorch that allows for tensor computation without actually performing it, useful for debugging and testing.
- **HPU**: Habana Processing Unit, specialized for AI and machine learning tasks.
- **MTIA**: Multi-threaded Interconnect Architecture, designed for high-efficiency data communication between devices.

Once the code is executed sucessfully you will find the "vector_database" folder created within the "data" folder.

# Executing the code for Command line interface
In order to chat using command line, run the following command (by default, it will run on `cuda`).
```shell
python run_localLLM.py
```
You can also specify the device type just like `createVectorDb.py`
```shell
python run_localLLM.py --device_type mps # to run on Apple silicon
```

## Extra Options with run_localLLM.py

You can use the `--show_sources` flag with `run_localLLM.py` to show which chunks were retrieved by the embedding model. Default is not to show the sources
```shell
python run_localLLM.py --show_sources
```

Another option is to enable chat history. Default is not to save history
```shell
python run_localLLM.py --use_history
```

Another option is to save the chat in a csv file. Default is not to save
```shell
python run_localLLM.py --save_qa
```

Another option is to sselect the model type ["llama", "mistral", "non_llama"]. Default is "llama"
```shell
python run_localLLM.py --model_type 
```

Once the code is excecuted, you will be prompted to enter the company number you want to know about, type the company number, you can only type the company number that is available in the `data/vector_database` folder

# Executing the code for GUI (Streamlit)
In order to chat using the GUI, run the following command.
```shell
streamlit run run_localLLM.py
```
From the side menu, select the company you want to know about
Tick the options on the menu to select the device type, sources, history, save chat

## Steps to Switch Between Different LLM Models

To alternate between various LLM models, you need to adjust both the `MODEL_ID` and `MODEL_BASENAME` settings.

1. Locate and open `constants.py` in your preferred text editor.
2. Modify the values of `MODEL_ID` and `MODEL_BASENAME`. Note that for quantized models like `GGML`, `GPTQ`, `GGUF`, it's necessary to specify `MODEL_BASENAME`. For non-quantized models, set `MODEL_BASENAME` to `NONE`.
3. Several example models from HuggingFace are compatible and have been verified to work with this setup. These include original trained models (identified by the suffix HF or a .bin file in "Files and versions") and quantized models (identified by GPTQ suffix or having .no-act-order or .safetensors in "Files and versions").
4. For models with an HF suffix or a .bin file on their HuggingFace page:
   - Choose a `MODEL_ID`. For instance, `MODEL_ID = "TheBloke/guanaco-7B-HF"`.
   - Visit the corresponding [HuggingFace Repository](https://huggingface.co/TheBloke/guanaco-7B-HF).
5. For models labeled with GPTQ or having .no-act-order or .safetensors in their HuggingFace "Files and versions":
   - Select a `MODEL_ID`, like `model_id = "TheBloke/wizardLM-7B-GPTQ"`.
   - Navigate to the appropriate [HuggingFace Repository](https://huggingface.co/TheBloke/wizardLM-7B-GPTQ) and view "Files and versions".
   - Choose a model name and set it as `MODEL_BASENAME`, e.g., `MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"`.
6. Repeat similar steps for selecting `GGUF` and `GGML` models.

## GPU and VRAM Requirements

Below is the VRAM requirement for different models depending on their size (Billions of parameters). The estimates in the table does not include VRAM used by the Embedding models - which use an additional 2GB-7GB of VRAM depending on the model.

| Mode Size (B) | float32   | float16   | GPTQ 8bit      | GPTQ 4bit          |
| ------- | --------- | --------- | -------------- | ------------------ |
| 7B      | 28 GB     | 14 GB     | 7 GB - 9 GB    | 3.5 GB - 5 GB      |
| 13B     | 52 GB     | 26 GB     | 13 GB - 15 GB  | 6.5 GB - 8 GB      |
| 32B     | 130 GB    | 65 GB     | 32.5 GB - 35 GB| 16.25 GB - 19 GB   |
| 65B     | 260.8 GB  | 130.4 GB  | 65.2 GB - 67 GB| 32.6 GB - 35 GB    |


## Pending Work
- Implementing RAG for the streamlit application
- Getting the response in structured JSON format
- Providing the option for the user to switch the response between JSON format and regular text.
