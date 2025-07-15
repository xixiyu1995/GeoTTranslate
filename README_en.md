# GeoTTranslate: GeoTTranslate: A LLM Framework for Geological Term Translation to Enhance Terminology Accuracy

@article{yu2022geotermx,
  title={GeoTTranslate: GeoTTranslate: A LLM Framework for Geological Term Translation to Enhance Terminology Accuracy},
  author={Xinhui Yu },
  email={yuxh29@mail2.sysu.edu.cn}
}



# GeoTTranslate Project

**GeoTTranslate** is a localized bidirectional Chinese-English translation framework designed specifically for the geological domain, addressing the shortcomings of general-purpose translation systems in terms of terminology accuracy and data privacy. The system integrates Geological Named Entity Recognition (GNER) with Retrieval-Augmented Generation (RAG) technology, leveraging local inference models and vector databases to achieve terminology-enhanced translation, data privacy protection, and high semantic consistency.

GeoTTranslate is built upon the Qwen-3 model, with the GNER module fine-tuned using LoRA (LoRA-GNER) to deliver high-precision terminology recognition in structured JSON format. The semantic retrieval component utilizes Milvus to construct a terminology vector database for dynamic domain knowledge injection. The translation module employs a locally deployed DeepSeek-R1 model, ensuring no cloud-based data transmission and safeguarding sensitive texts (e.g., international exploration reports).

## Project Highlights

Experiments on geological corpora (10,803 Chinese sentences and 31,943 English sentences) demonstrate:

- **Named Entity Recognition Performance**: LoRA-GNER achieves F1 scores of 0.987 for English and 0.938 for Chinese, outperforming models like BERT-BiLSTM-CRF and BERT-IDCNN-CRF.
- **Terminology Consistency Improvement**: Terminology Accuracy (TA) reaches 73.4% (Chinese to English) and 76.4% (English to Chinese), significantly surpassing the baseline DeepSeek-R1 system.
- **Error Rate Reduction**: Mistranslation and omission rates are reduced by over 40% and 50%, respectively.
- **Semantic Fidelity Improvement**: BERT-Score F1 improves by 4.2% and 3.8%.

GeoTTranslate exhibits strong robustness in terminology accuracy, semantic consistency, and data privacy, providing a controllable and efficient translation solution for geology and other highly specialized domains.

## Project Structure

The project includes the following main modules:

```
GeoTTranslate/
├── BERTScore/                        # BERT-Score evaluation scripts and model configurations
├── GeoTermX_translation/             # Translation module (supports RAG and DeepSeek methods)
│   ├── 100_en.txt / 100_zh.txt       # Sample English/Chinese input corpora
│   ├── Trans-RAG-.py                 # Bidirectional translation using RAG
│   ├── Trans-Deepseek-.py            # Bidirectional translation using DeepSeek local model
│   └── *.csv                         # Translation result outputs
├── GNER/                             # Geological Named Entity Recognition (GNER) module
│   ├── GNER_LoRA/                    # LoRA fine-tuning module and prediction logic
│   │   ├── en/ zh/                   # English/Chinese entity recognition experiment configs and data
│   │   ├── train.py / test.py        # Fine-tuning and inference scripts
│   │   └── utils.py                  # Utility functions
│   └── NER_bert-bilstm_idcnn-crf/    # Baseline NER models (e.g., BiLSTM, IDCNN)
├── Milvus_database/                  # Vector database module (terminology injection)
│   ├── dictdata/                     # Dictionary files and terminology vector data
│   ├── milvus_connect.py             # Vector database connection management
│   └── Milvus_DICTdata.py            # Vector construction and retrieval interface
├── requirements.txt                  # Project dependency list
└── README_zh.md / README_en.md       # Chinese/English documentation
```

## Environment Requirements

- **Python Version**: 3.10
- **Recommended Environment**: Use a virtual environment manager (e.g., `venv` or `conda`)
- **Deep Learning Framework**: PyTorch 2.5.1 (CUDA 12.1)
- **Hardware Requirements**:
  - Recommended: NVIDIA GPU (supporting CUDA 12.1)
  - System Memory: 32GB+ recommended
  - GPU Memory: ≥ 12GB (e.g., RTX 3080/3090)

## Installation Instructions

### Create and Activate Virtual Environment

**Using venv**:

```bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows
```

**Or using Conda**:

```bash
conda create -n geotermx python=3.10 -y
conda activate geotermx
```

### Install Dependencies

Install standard dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

The project modules can be run independently. Follow these steps for the complete workflow:

### Step 1: Build Milvus Vector Terminology Database

Used for terminology-level semantic enhancement. Requires a running Milvus database.

```bash
cd Milvus_database
python milvus_connect.py      # Start and connect to local Milvus service
python Milvus_DICTdata.py     # Load terminology dictionary and populate vector database
```

⚠️ **Note**: Milvus must be deployed locally, preferably using Docker (Milvus standalone).

### Step 2: Perform Geological Entity Recognition (GNER)

Train the GNER module using LoRA fine-tuning. Default configuration is for English recognition.

```bash
cd GNER/GNER_LoRA
python train.py    # Fine-tune LoRA-GNER model
python test.py     # Perform inference and output entities in JSON format
```

📁 For Chinese, switch to the `GNER/GNER_LoRA/zh/` directory and adjust configurations.

### Step 3: Perform Terminology-Enhanced Translation

Two translation methods are provided:

#### 3.1 Using RAG Translation

```bash
cd Geo-translation
python Trans-RAG-en-to-zh.py
python Trans-RAG-zh-to-en.py
```

#### 3.2 Using DeepSeek-R1 Model Translation

```bash
python Trans-Deepseek-en-to-zh.py
python Trans-Deepseek-zh-to-en.py
```

⚠️ **Note**: DeepSeek-R1 and Qwen-3 models must be downloaded locally and their paths specified in the scripts.

### Step 4: Evaluate Translation Quality (Optional)

Use BERT-Score to assess the semantic fidelity of translation results.

## Model Preparation

The project requires the following locally deployed models, which must be downloaded in advance:

- **Qwen-3**: Used for GNER tasks (HuggingFace format recommended)
- **DeepSeek-R1**: Used for local inference translation (supports FP16 inference)
- **Optional**: Custom terminology embedding models (e.g., BERT)

After downloading, update the `model_path` parameters in relevant `.py` scripts.

## Model and Data Description

### Model Components

| Module              | Model Used         | Description                                              |
|---------------------|--------------------|----------------------------------------------------------|
| GNER Entity Recognition | Qwen-3 + LoRA      | LoRA-fine-tuned Qwen-3 model with structured JSON output |
| Translation Model   | DeepSeek-R1        | Locally deployed translation model ensuring data privacy  |
| Vector Retrieval    | Milvus + Embedding | Milvus-based semantic terminology retrieval system       |
| Quality Evaluation  | BERT-Score         | Semantic similarity evaluation using F1 score            |

#### Chinese Geological Entity Recognition Data (GNER)

- **Source**: Ma Kai et al. (2022) public geological entity recognition dataset
- **Corpus Construction**: Based on four regional geological survey reports (Nima, Zhiduo, Jinniuzhen-Gaoqiao, Yangchun)
- **Link**: [https://www.sciengine.com/JGCDD/doi/10.3974/geodp.2022.01.11](https://www.sciengine.com/JGCDD/doi/10.3974/geodp.2022.01.11)
- **Data Scale**: 10,803 sentences
- **Annotation Strategy**:
  - Removed descriptive adjectives (e.g., color, grain size)
  - Enhanced independent entity annotation (e.g., "Geren volcanic rock" → "Geren—Location LOC" + "Volcanic rock—Rock ROCK")
- **Purpose**: Train the LoRA-GNER model

#### English Mineral Named Entity Recognition Data (OzROCK)

- **Source**: [OzROCK Open-Source Project](https://github.com/majiga/OzROCK)
- **Scale**: 31,943 English sentences
- **Purpose**: Train the English GNER model (LoRA-GNER)

### Model Download Instructions

The following models must be downloaded locally:

- **Qwen-3 (Entity Recognition)**: Recommended from [HuggingFace](https://huggingface.co); specify path in `train.py`.
- **DeepSeek-R1 (Translation)**: Must support local FP16 inference.
- **BGE Embedding Model (Optional)**: For terminology vectorization (e.g., `bert-base`).

## Evaluation Metrics and Experimental Results

The project evaluates the LoRA-GNER model’s performance on geological named entity recognition tasks, comparing it with mainstream deep learning methods (e.g., BERT-BiLSTM-CRF, BERT-IDCNN-CRF).

### Entity Recognition Performance (F1-Score)

| Model                 | Chinese (Coarse) | Chinese (Fine) | English   |
|-----------------------|------------------|----------------|-----------|
| **LoRA-GNER**         | 0.693            | 0.938          | **0.987** |
| BERT-BiLSTM-CRF       | 0.671            | 0.930          | 0.972     |
| BERT-IDCNN-CRF        | 0.640            | 0.946          | 0.960     |

> LoRA-GNER excels in English entity recognition and performs near-optimally in fine-grained Chinese recognition. Its structured JSON output is well-suited for downstream terminology enhancement and translation tasks.

For translation evaluation, GeoTTranslate significantly improves Terminology Accuracy (TA) and reduces Mistranslation Rate (MR) and Omission Rate (OR). Detailed results are provided in the project paper.
