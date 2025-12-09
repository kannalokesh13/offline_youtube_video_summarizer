## Project Description:

This application takes a public YouTube video URL and automatically generates a concise summary of its content. It works by downloading the video’s audio, converting the speech to text using an offline transcription model, and then using an offline language model to summarize the transcribed text. The final summary is displayed to the user without requiring any cloud-based services.


## Setup and Installation Instructions:
- First download the above zip file and extract it, there you can find all the project structure. And open this project and then follow the instructions in the README.md file.

- Create a new virtual Environment and make sure the python version is 3.10
    ```bash
    conda create -n vene python=3.10
    ```

- Activate the conda environment and make sure that the version version in 3.10
    ```bash
    conda activate vene
    ```

- Install the libraries using the below command
    ```bash
    pip install -r requirements.txt
    ```

- Install the llama-cpp-python library tool using the below code if you are ruunig this application in windows or if llama-cpp-python failed to install
    ```bash
    pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.2/llama_cpp_python-0.3.2-cp310-cp310-win_amd64.whl
    ```

- Once you installed all the libraries, Download the open source llm model, Here i used the Llama-3.1-8B-instruct-Q4_K_M.gguf using the below link and place it in Llama_model Folder for text summarization.
    https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/tree/main

- Download the Whisper Model if it not there using the below code and then place it in the wishper_model Folder for speech to text recognition. Copy the "base.pt" file into wishper folder.
    - Install the whisper library
    ```bash
    pip install openai-whisper
    ```
    - Download the Model
    ```python
    import whisper
    model = whisper.load_model("base", download_root="wishper_model/")
    ```

- we could not able to download any Youtube video, so to download any video we need to pass the cookies.txt file in this project folder. To create the cookies.txt file follow the below approch. (The cookie file will expire after some time and then replace the existing cookie file with new one with below process)
    - Download the "Get cookies.txt Locally" extension in your browser and add this extension.
    - Open https://youtube.com and then login to your YouTube account and then at right top click on the get cookies extension and the click on Export and place that file in this project folder.
    - Make sure that name of that text file file is "cookie.txt"

- Once the above all steps were completed, then strat this application using the below command 
```python
python app.py
```

- Copy and Paste the localhost address "http://127.0.0.1:5000" in the browser URL and the provide the Https YouTube URL and then click submit, it will provide the title for that video and the summary of that video.


## Design Choices and Justification:

1. Speech-to-Text Model: Whisper (Offline Transcription)

    - I selected OpenAI Whisper for converting the YouTube audio into text because of its strong balance between accuracy, robustness, and offline usability. Whisper is known for handling diverse audio conditions accents, background noise, and varying speech speeds better than many traditional speech-to-text models.

    - Larger Whisper models (large-v2, large-v3) provide higher accuracy but require more compute and memory.
    Smaller models (tiny, base) are faster but less accurate.

    - I chose a model size that balances accuracy and performance for local execution without needing a GPU.

2. Text Summarization Model: Llama-3.1-8B-Instruct (Q4_K_M.gguf)

    - For summarization, I selected Meta Llama-3.1-8B-Instruct in GGUF Q4_K_M quantized format. This model provides strong reasoning and summarization abilities while remaining small enough to run efficiently on local hardware through llama.cpp.
    High-quality summaries compared to older open-source models like GPT-J, GPT-NeoX, or smaller Llama variants.
    Instruction-tuned, which means it follows summarization prompts reliably without extra fine-tuning.

    - Larger models (Llama-3-70B, Mixtral 8x7B) provide better reasoning but are too heavy for typical local machines.Smaller models (Llama-3-3B, Mistral-7B) run faster but sometimes produce weaker or overly generic summaries.

    - Choosing the 8B model in Q4_K_M format provides a sweet spot between performance, cost, and summarization quality.

## Usage:

- Once all the setup has been doen mentioned in the Setup and Installation Instructions section, then it will open a web page where you can pass the YouTube URL, it will take a little bit to provide the output, then it will provide the summary of  that video.
```python 
python app.py
```

OR 

- In the ReSearch Folder there is a .ipynb file where you can run the all cells mentioned in the description there, it will provide the output, you can run this in google colab whithout any extra efforts, but make sure the models were downloaded and kept them in spcified folder in the .ipynb file.

## Challenges Faced:

- I am develping this application in windows where i faced some issues related to the installation of llama-cpp-python library becuase windows doesn't support the cmake commands.

- Selecting the proper summarisation model, where many open source models were there but all of them are not providing the output how we intended and also prompting for this model.

- Faced issues related to the ffmpeg tools while trying to download the youtube video or extracting the audio from that and while performing infernce upon the whisper model.

- Making this application to download all kinds of youtube videos.


## Demo Video:
Here I have provided the Drive Link for that Video
[DEMO VIDEO](https://drive.google.com/file/d/1LQGWQoLQOylgN22VpOVIz2nyiZIthG0C/view?usp=sharing)










    
