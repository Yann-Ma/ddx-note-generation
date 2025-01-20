# DDX-note-generation
Source code of the research on differential diagnostic (DDX) note generation.

### 1 Python Environment Requirements

- python>=3.8
- transformers==4.44.0
- tiktoken
- torch
- pandas
- openpyxl
- gensim
- rouge-chinese
- bert-score
- nltk
- jieba
- tqdm
- openai
- numpy

### 2 Code Structure

- **GENERATION** The files `generate_response_onLLM.ipynb` and `generate_response_offLLM.py` are used to prompt online and offline LLMs to generate differential diagnostic notes. Note that `generate_response_offLLM.py` uses an available GPU (an NVIDIA GTX 3090 GPU in our experiments) to prompt the offline LLM `GLM-4-9b` to generate differential diagnostic notes.

  ~~~bash
  python generate_response_onLLM.py
  ~~~

  The execution of `generate_response_offLLM.ipynb` has no limis to machines while it requires an available **OpenAI API key**.

- **EVALUATION** The generated notes are evaluated in `evaluation_allLLM.ipynb`. The datasets are analyzed in `dataset_stat.ipynb`.

### 3 Instructions to Reproduce the Codes

1. **Model preparations.**

   1. Prepare an available OpenAI API key.
   2. Download the model [GLM-4-9b](https://huggingface.co/THUDM/glm-4-9b) and [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) from Huggingface to the folder `./models`.  (If you want to load bert-base-chinese from your local path, you should modify the codes of `class BERTScore` in the bert-score package)
   3. Download the trained skip-gram (word embedding) model from [Google Drive](https://drive.google.com/drive/folders/1GnMyB1gxei_0ZZnhPND0wTnbhoT2g6nQ?usp=sharing) or [BaiduNetDisk](https://pan.baidu.com/s/1P0OpYROFtJsNRYi0n_8MNA?pwd=cpbj) to the folder `./models`. 
   4. Modify the model paths in the programs.

2. **Data preparations.**

   1. Download [data]( https://dx.doi.org/10.21227/1t6e-rd05).
   2. Modify the dataset paths in the programs.

3. **Generation.** Execute the cells in `generate_response_onLLM.ipynb`.

   *Notice:* the generation takes much time and requires paid api. For convenience, we provide the generation results in the folder `./output/` so that you can directly execute the evaluation codes.

4. **Evaluation.**

   1. The results of LLM response evaluation are shown in the file `evaluation_allLLM.ipynb`. You can also execute the cells in it.
   2. Data statistics are shown in the file `dataset_stat.ipynb`. You can also execute the cells in it to analyze the dataset.



### 4 More Details

For more details about the functions called during generation and evaluation, you can read the files in the folder `./utils`. 

- The file `eval.py` contains all the evaluation metrics in the experiments.
- The file `generate_gpt.py` load the dataset and provides the interfaces to call online GPT models.
- The file `generate_glm_cuda.py` is similar to `genearte_gpt.py`, but it should be executed on the machine with GPU and the GLM-4-9b model.
