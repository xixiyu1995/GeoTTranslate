# GeoTTranslate: GeoTTranslate: A LLM Framework for Geological Term Translation to Enhance Terminology Accuracy

@article{yu2022geotermx,
  title={GeoTTranslate: GeoTTranslate: A LLM Framework for Geological Term Translation to Enhance Terminology Accuracy},
  author={Xinhui Yu },
  email={yuxh29@mail2.sysu.edu.cn}
}



# GeoTTranslate 项目

**GeoTTranslate** 是一个专为地质领域设计的本地化中英文双向翻译框架，解决了通用翻译系统在术语精准性与数据隐私方面的不足。该系统集成了地质命名实体识别（GNER）与检索增强生成（RAG）技术，结合本地推理模型与向量数据库，实现术语增强翻译、隐私保护与高语义一致性。

GeoTTranslate 基于 Qwen-3 模型，通过 LoRA 技术微调 GNER 模块（LoRA-GNER），以结构化 JSON 格式实现高精度术语识别。语义检索部分采用 Milvus 构建术语向量库，动态注入领域知识；翻译模块采用本地部署的 DeepSeek-R1 模型，避免任何云端数据传输，保障敏感文本（如国外勘探报告）不出本地。

## 项目亮点

在地质中英文语料（中文 10,803 句，英文 31,943 句）上的实验证明：

- **命名实体识别性能**：LoRA-GNER 在英文/中文文本上的 F1 分别达到 0.987 和 0.938，优于 BERT-BiLSTM-CRF 和 BERT-IDCNN-CRF 等模型。
- **术语准确性提升**：TA 达到 73.4%（中译英）和 76.4%（英译中），显著优于单独使用 DeepSeek-R1 的基线系统。
- **错误率降低**：误译率与遗漏率分别降低超 40% 和 50%。
- **语义保真度提升**：BERT-Score F1 提升 4.2% 和 3.8%。

GeoTTranslate 在术语准确性、语义一致性与数据隐私方面均表现出强鲁棒性，为地质及其他高专业性领域提供了可控、高效的翻译解决方案。

## 项目结构

本项目包含以下主要模块：

```
GeoTTranslate/
├── BERTScore/                        # BERT-Score 评估脚本与模型配置
├── GeoTermX_translation/             # 翻译主模块（含 RAG 与 DeepSeek 两种翻译方案）
│   ├── 100_en.txt / 100_zh.txt       # 示例英文/中文输入语料
│   ├── Trans-RAG-.py                 # 使用 RAG 方法进行中英文双向翻译
│   ├── Trans-Deepseek-.py            # 使用 DeepSeek 本地模型进行双向翻译
│   └── *.csv                         # 对应翻译结果输出
├── GNER/                             # 命名实体识别模块（GNER）
│   ├── GNER_LoRA/                    # LoRA 微调模块及预测逻辑
│   │   ├── en/ zh/                   # 中英文实体识别实验配置及数据
│   │   ├── train.py / test.py        # 微调与推理脚本
│   │   └── utils.py                  # 公共函数
│   └── NER_bert-bilstm_idcnn-crf/    # 其他NER基线模型（如 BiLSTM、IDCNN 等）
├── Milvus_database/                  # 向量数据库模块（术语注入）
│   ├── dictdata/                     # 字典文件与术语向量数据
│   ├── milvus_connect.py             # 向量数据库连接管理
│   └── Milvus_DICTdata.py            # 向量构建与检索接口
├── requirements.txt                  # 项目依赖列表
└── README_zh.md / README_en.md       # 中英文使用说明文档
```

## 环境要求

- **Python 版本**：3.10
- **推荐环境**：使用虚拟环境管理工具（如 `venv` 或 `conda`）
- **深度学习框架**：PyTorch 2.5.1（CUDA 12.1）
- **硬件要求**：
  - 推荐使用 NVIDIA GPU（支持 CUDA 12.1）
  - 系统内存推荐 32GB+
  - 显存建议 ≥ 12GB（如 RTX 3080/3090）

## 安装说明

### 创建并激活虚拟环境

**使用 venv：**

```bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows
```

**或使用 Conda：**

```bash
conda create -n GeoTTranslate python=3.10 -y
conda activate GeoTTranslate
```

### 安装依赖项

首先安装标准依赖项：

```bash
pip install -r requirements.txt
```

## 快速开始

本项目各模块可独立运行，建议按以下顺序执行以完成完整流程：

### 步骤 1：构建 Milvus 向量术语库

用于术语级语义增强，需提前运行 Milvus 数据库。

```bash
cd Milvus_database
python milvus_connect.py      # 启动并连接本地 Milvus 服务
python Milvus_DICTdata.py     # 加载术语词典并写入向量库
```

⚠️ **注意**：需本地已部署 Milvus，推荐使用 docker 部署 Milvus standalone。

### 步骤 2：执行地质实体识别（GNER）

以 LoRA 微调方式训练 GNER 模块。默认配置为英文识别。

```bash
cd GNER/GNER_LoRA
python train.py    # 微调 LoRA-GNER 模型
python test.py     # 推理并输出 JSON 格式实体结果
```

📁 中文路径为 `GNER/GNER_LoRA/zh/`，可自行切换数据与配置。

### 步骤 3：执行术语增强翻译

提供两种翻译方式：

#### 3.1 使用 RAG 翻译

```bash
cd Geo-translation
python Trans-RAG-en-to-zh.py
python Trans-RAG-zh-to-en.py
```

#### 3.2 使用 DeepSeek-R1 模型翻译

```bash
python Trans-Deepseek-en-to-zh.py
python Trans-Deepseek-zh-to-en.py
```

⚠️ **注意**：DeepSeek-R1 与 Qwen-3 模型需用户提前下载至本地，并在脚本中指定路径。

### 步骤 4：评估翻译质量（可选）

使用 BERT-Score 指标评估翻译结果的语义保真度。


## 模型准备

本项目需以下本地模型支持，使用前请提前下载：

- **Qwen-3**：用于 GNER 识别任务（推荐 HuggingFace 格式）
- **DeepSeek-R1**：用于本地推理翻译（需支持 FP16 推理）
- **可选**：自定义术语嵌入词向量模型（如 bert）

模型下载后，请根据路径修改相关 `.py` 脚本中的 `model_path` 参数。

## 模型与数据说明

### 模型组件

| 模块        | 使用模型          | 说明                                  |
|-------------|-------------------|----------------------------------------|
| GNER 实体识别 | Qwen-3 + LoRA     | 基于 Qwen-3 的 LoRA 微调模型，实现结构化输出（JSON） |
| 翻译模型     | DeepSeek-R1       | 本地部署翻译模型，保障数据私密性（无云端调用）       |
| 向量检索     | Milvus + Embedding | 利用 Milvus 构建领域术语语义检索系统         |
| 质量评估     | BERT-Score        | 语义相似度评估翻译质量，采用 F1 分数衡量         |

#### 中文地质实体识别数据（GNER）

- **来源**：马凯等（2022）公开地质实体识别数据集
- **语料构建**：基于四份区域地质调查报告（尼玛区幅、治多县幅、金牛镇幅高桥幅、阳春县幅）
- https://www.sciengine.com/JGCDD/doi/10.3974/geodp.2022.01.11
- **数据规模**：共 10,803 句
- **标注策略**：
  - 删除修饰性形容词（如颜色、粒级等）
  - 强化独立实体标注（如“格仁火山岩” → “格仁—地点 LOC” + “火山岩—岩石 ROCK”）
- **用途**：训练 LoRA-GNER 模型

#### 英文矿产命名实体识别数据（OzROCK）

- **来源**：[OzROCK 开源项目](https://github.com/majiga/OzROCK)
- **规模**：共 31,943 个英文句子
- **用途**：训练英文 GNER 模型（LoRA-GNER）

### 模型下载说明

以下模型需用户手动下载至本地：

- **Qwen-3（实体识别）**：推荐使用 [HuggingFace](https://huggingface.co) 上的 Qwen 模型，加载路径需在 `train.py` 中指定。
- **DeepSeek-R1（翻译）**：需支持本地 FP16 推理Lithium Battery
- **BGE Embedding 模型（可选）**：用于术语向量化，如采用 `bert-base` 等模型。

## 评估指标与实验结果

本项目评估了 LoRA-GNER 模型在地质命名实体识别任务中的性能，并与主流深度学习方法（如 BERT-BiLSTM-CRF 和 BERT-IDCNN-CRF）进行了对比。

### 实体识别性能（F1-score）

| 模型                 | 中文（粗粒度） | 中文（细粒度） | 英文      |
|----------------------|----------------|----------------|-----------|
| **LoRA-GNER**        | 0.693          | 0.938          | **0.987** |
| BERT-BiLSTM-CRF      | 0.671          | 0.930          | 0.972     |
| BERT-IDCNN-CRF       | 0.640          | 0.946          | 0.960     |

> 其中，LoRA-GNER 在英文实体识别中表现最优，中文细粒度下接近最佳。其输出结构化 JSON 格式，适配后续术语增强与翻译模块。

翻译评估方面，GeoTTranslate 框架显著提升术语准确率（TA），并有效降低误译率（MR）与漏译率（OR）。详细实验数据见论文正文。



