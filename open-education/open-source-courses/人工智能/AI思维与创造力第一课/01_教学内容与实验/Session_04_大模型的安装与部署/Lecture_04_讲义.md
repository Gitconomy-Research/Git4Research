# Qwen 2.5 介绍与部署流程

## 前言

在国产大模型领域，Qwen 系列一直稳居前列，其出色的性能使其在多项评测中名列前茅。作为阿里巴巴的一项重要研发成果，Qwen 系列的开源版本在业内备受瞩目，且长期以来在各大榜单上表现优异。2024年9月，阿里重磅推出了全新升级的 Qwen2.5 系列模型，涵盖了不同参数规模的版本，以满足多样化的应用需求。此外，Qwen2.5 系列还推出了各具特色的强化版本，进一步提升了模型在特定任务上的表现。接下来，让我们一同深入了解 Qwen2.5 系列的具体技术特点及其在实际应用中的优势。

本课程旨在为学员提供关于Qwen 2.5系列模型的全面了解和实际部署技能。通过学习，您将能够：

* **理解Qwen 2.5模型概况：** 掌握Qwen 2.5系列模型的基本参数、最新升级特性（如更大规模数据集、知识储备、代码和数学能力增强）及其在不同规模下的性能表现。
* **掌握多种部署方法：**
    * **ModelScope本地部署：** 学会如何在本地创建Conda虚拟环境，安装PyTorch、Transformers及ModelScope等必要依赖，并使用脚本进行模型的下载、加载和运行测试。
    * **ModelScope SDK部署：** 理解ModelScope平台云端部署的优势，并掌握通过SDK调用Qwen 2.5模型进行推理的方法。
    * **Ollama框架部署：** 熟悉Ollama工具的基本信息、在Linux和Windows环境下安装与使用流程，包括模型的下载、运行聊天测试及文件管理。
    * **vLLM框架部署：** 了解vLLM框架在高吞吐量和低延迟推理方面的优势，并掌握其安装和使用vLLM进行模型推理部署的方法。

通过本课程的学习，学员将不仅对Qwen 2.5模型有深入的理论认识，更能获得在不同环境下实际部署和应用大模型的宝贵经验。

# 一、Qwen 2.5模型介绍

## 1.1 基本参数介绍

通义千问是由阿里巴巴的通义千问团队研发的一系列大规模语言和多模态模型。该模型能够执行多种任务，包括自然语言理解、文本生成、视觉理解、音频理解、工具调用、角色扮演和智能体操作等。语言和多模态模型均在大规模、多语言和多模态的数据上进行预训练，并在高质量语料上进行后续训练，以使其与人类的偏好保持一致。同时发布开源和闭源两大版本。

2024年9月19日最新发布的模型包括语言模型 Qwen2.5，这个系列涵盖了多种尺寸的大语言模型、多模态模型、数学模型以及代码模型，构建了一个完善的模型体系，能够为不同领域的应用提供强有力的支持。不论是在自然语言处理任务中的文本生成与问答，还是在编程领域的代码生成与辅助，或是数学问题的求解，Qwen2.5 都能展现出色的表现。每种尺寸的模型均包含基础版本、指令跟随版本和量化版本，共推出了100多个模型，充分满足了用户在各类应用场景中的多样化需求。具体版本内容如下：

  - Qwen2.5: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 以及72B;
  - Qwen2.5-Coder: 1.5B, 7B, 以及即将推出的32B;
  - Qwen2.5-Math: 1.5B, 7B, 以及72B。

相比于 Qwen2 系列，Qwen2.5 带来了以下全新升级：

  - 全面开源：考虑到用户对 10B 至 30B 范围模型的需求以及移动端对 3B 模型的兴趣，此次不仅继续开源 Qwen2 系列中的 0.5B/1.5B/7B/72B 四款模型，Qwen2.5 系列还新增了两个高性价比的中等规模模型—— Qwen2.5-14B 和 Qwen2.5-32B，以及一款适合移动端的 Qwen2.5-3B。所有模型在同类开源产品中均具有较强的竞争力，例如 Qwen2.5-32B 的整体表现超越了 Qwen2-72B，而 Qwen2.5-14B 则领先于 Qwen2-57B-A14B。

  - 更大规模、更高质量的预训练数据集：预训练数据集的规模从 7T tokens 扩展至 18T tokens。

  - 知识储备升级：Qwen2.5 的知识涵盖面更广。在 MMLU 基准测试中，Qwen2.5-7B 和 72B 的得分分别从 Qwen2 的 70.3 提升至 74.2，以及从 84.2 提升至 86.1。此外，Qwen2.5 在 GPQA、MMLU-Pro、MMLU-redux 和 ARC-c 等多个基准测试中也有明显提升。

  - 代码能力增强：得益于 Qwen2.5-Coder 的突破，Qwen2.5 在代码生成能力上大幅提升。Qwen2.5-72B-Instruct 在 LiveCodeBench（2305-2409）、MultiPL-E 和 MBPP 中的得分分别为 55.5、75.1 和 88.2，明显优于 Qwen2-72B-Instruct 的 32.2、69.2 和 80.2。

  - 数学能力提升：引入了 Qwen2-math 的技术后，Qwen2.5 在数学推理表现上也有快速提升。在 MATH 基准测试中，Qwen2.5-7B/72B-Instruct 的得分分别从 Qwen2-7B/72B-Instruct 的 52.9/69.0 上升至 75.5/83.1。

  - 更符合人类偏好：Qwen2.5 生成的内容更贴近人类的偏好。具体来说，Qwen2.5-72B-Instruct 在 Arena-Hard 测试中的得分从 48.1 大幅提升至 81.2，而 MT-Bench 的得分也从 9.12 提升至 9.35，相较于之前的 Qwen2-72B 有显著提升。

  - 其他核心能力提升：Qwen2.5 在指令跟随、生成长文本（从 1K 升级到 8K tokens）、理解结构化数据（如表格）以及生成结构化输出（特别是 JSON）方面都有明显进步。此外，Qwen2.5 能够更好地响应多样化的系统提示，支持用户为模型设置特定角色或自定义条件。

就 Qwen2.5 语言模型而言，所有模型均在最新的大规模数据集上进行了预训练，该数据集包含多达 18T tokens。与 Qwen2 相比，Qwen2.5 获得了显著更多的知识（MMLU：85+），并在编程能力（HumanEval 85+）和数学能力（MATH 80+）方面实现了大幅提升。此外，新模型在指令执行、生成长文本（超过 8K tokens）、理解结构化数据（如表格）以及生成结构化输出，特别是 JSON 方面也取得了显著进展。总体而言，Qwen2.5 模型对各种系统提示表现出更强的适应性，增强了角色扮演的实现和聊天机器人的条件设置功能。与 Qwen2 相似，Qwen2.5 语言模型支持高达 128K tokens，并能够生成最多 8K tokens 的内容，同时保持对包括中文、英文、法文、西班牙文、葡萄牙文、德文、意大利文、俄文、日文、韩文、越南文、泰文、阿拉伯文等在内的 29 种以上语言的支持。关于模型的基本信息已在下表中提供。

专业领域的专家语言模型，如用于编程的 Qwen2.5-Coder 和用于数学的 Qwen2.5-Math，相较于前身 CodeQwen1.5 和 Qwen2-Math 实现了实质性改进。具体来说，Qwen2.5-Coder 在包含 5.5T tokens 编程相关数据上进行了训练，使得即使是较小的编程专用模型在编程评估基准测试中也能表现出与大型语言模型相媲美的竞争力。同时，Qwen2.5-Math 支持中文和英文，并整合了多种推理方法，包括链式推理（CoT）、程序推理（PoT）和工具集成推理（TIR）。

Qwen2.5 的一项重要更新是重新推出了 Qwen2.5-14B 和 Qwen2.5-32B。这些模型在各种任务中表现优异，超越了同等或更大规模的基线模型，如 Phi-3.5-MoE-Instruct 和 Gemma2-27B-IT。Qwen2.5 系列在模型规模和能力之间取得了良好平衡，提供了与一些更大型模型相当甚至更优的性能。

近年来，小型语言模型（SLMs）出现了明显的转向趋势。尽管历史上小型语言模型的表现一直落后于大型语言模型（LLMs），但二者之间的性能差距正在迅速缩小。值得注意的是，即使是只有约 30 亿参数的模型，现在也能够取得高度竞争力的结果。附带的图表显示了一个重要的趋势：在 MMLU 中得分超过 65 的新型模型正逐渐变得更小，这凸显了语言模型知识密度增长速度的加快。特别值得一提的是，Qwen2.5-3B 成为这一趋势的典型例子，凭借约 30 亿参数实现了令人印象深刻的性能，展现了相较于前辈模型的高效性和能力。

Qwen2.5 还推出了专门针对数学问题的模型——Qwen2.5-Math。其数学能力得分十分亮眼，这得益于其大规模的训练数据和专门的数学模型设计。与其他顶尖模型相比，Qwen2.5 不仅在数学推理上表现优异，还在编程能力上展现出强大的竞争力。该模型在数学推理方面进行了特别优化，支持中文和英文，并整合了多种推理方法，包括：思维链（Chain of Thought, CoT）：帮助模型在解决复杂问题时进行逐步推理。工具集成推理（Tool-Integrated Reasoning, TIR）：增强模型在解决数学问题时的灵活性和准确性。

Qwen2.5-Math-72B-Instruct 的整体性能超越了 Qwen2-Math-72B-Instruct 和 GPT4-o，甚至是非常小的专业模型如 Qwen2.5-Math-1.5B-Instruct 也能在与大型语言模型的竞争中取得高度竞争力的表现。

**Qwen模型版本代号系统说明**

在Qwen大模型的版本命名中，您会看到一系列代号（或后缀），这些代号提供了关于模型特性、优化方式或文件格式的关键信息，帮助用户理解模型的用途和部署要求。以下是这些常见代号的系统说明：

#### 1. Instruct (指令微调版本)

* **含义：** 代表 **Instruction Fine-tuning (指令微调)**。
* **功能与目的：** 这种模型在基础预训练之后，经过了专门的指令遵循训练（通常通过监督式微调 SFT 或人类反馈强化学习 RLHF）。其目标是让模型能够更好地理解和执行用户的自然语言指令，并生成符合人类偏好、有用且无害的回答。
* **应用场景：** 最适合直接用于构建智能聊天机器人、虚拟助手、问答系统以及其他需要模型精确响应用户命令的交互式AI应用。

#### 2. AWQ (激活感知权重量化)

* **含义：** 代表 **Activation-aware Weight Quantization (激活感知权重量化)**。
* **功能与原理：** 是一种高效的**后训练量化（PTQ）**方法。AWQ 的核心在于它能够识别模型中对性能影响最关键的权重，并对其进行保护性量化，而对其他相对不那么敏感的权重进行更激进的低比特量化（如 INT4）。
* **优势：** 在大幅减小模型体积和加速推理的同时，能够最大程度地保持模型精度，通常优于简单的统一量化方法。
* **应用场景：** 适用于在资源受限环境（如边缘设备、消费级GPU）或需要高吞吐量的推理服务中部署大模型。

#### 3. GPTQ (生成式预训练Transformer量化)

* **含义：** 代表 **Generative Pre-trained Transformer Quantization**。
* **功能与原理：** 也是一种先进的**后训练量化（PTQ）**方法。GPTQ 的特点是“一次性”（one-shot）量化，它通过优化算法寻找最佳的低比特量化配置，通常只需要一小部分校准数据（甚至一个批次），就能在极低比特位（如 INT4）下实现较高的精度。
* **优势：** 量化过程快速高效，能在极低的比特精度下保持出色的模型性能，无需额外训练。
* **应用场景：** 与 AWQ 类似，用于在资源有限或追求极致推理速度的场景下部署高效的量化模型。

#### 4. Int4 (4比特整数表示)

* **含义：** 指将模型的权重和/或激活值量化为 **4比特整数**进行存储和计算。
* **功能与特点：**
    * **极致压缩：** 将模型文件大小显著减小（理论上是FP16模型的1/4，FP32模型的1/8），大幅降低存储和传输成本。
    * **推理加速：** 4比特运算效率高，能显著提升推理速度，尤其是在支持低比特优化的硬件上。
    * **与量化方法的关系：** Int4 是量化所达到的**精度级别**。AWQ 和 GPTQ 则是实现这种 Int4 量化的**具体方法或算法**。
* **应用场景：** 专为在硬件资源极其有限的环境中（如移动设备、嵌入式系统）部署大型模型，或追求最高推理吞吐量的场景。

#### 5. GGUF (GPT-Generated Unified Format)

* **含义：** 代表 **GPT-Generated Unified Format** (通常是GGML/GGMF格式的继任者)。
* **功能与特点：** GGUF 是一种**文件格式**，专为存储和分发LLM而设计，尤其优化了在CPU上的本地推理。它是一个单一、自包含的文件，包含模型权重、词汇表和元数据，并支持多种量化级别（如 Q2_K, Q4_K, Q5_K 等）。
* **优势：**
    * **高度便携：** 单一文件易于管理和分发。
    * **CPU优化：** 通过内存映射技术实现快速加载和低内存占用。
    * **社区标准：** 广泛应用于 `llama.cpp` 等项目，成为本地LLM推理的事实标准。
* **与量化方法的关系：** GGUF 是一种**容器格式**。经过 AWQ、GPTQ 等方法量化（如量化到 Int4 精度）后的模型，通常会被封装成 GGUF 文件，以便于本地部署和运行。

通过理解这些代号，用户可以根据模型的功能需求（如是否需要指令遵循）、部署环境的资源限制（如是否需要极度压缩）以及对推理性能的要求，更精准地选择和配置Qwen大模型的相应版本。

  - 更多信息可以访问Qwen官网：[QwenLM Github](https://qwenlm.github.io/zh/blog/qwen2.5-llm/)

### 1.2 线上体验办法

在 Hugging Face 平台上，用户可以方便地进行该模型的线上体验。这种在线测试不仅可以帮助用户直观地了解模型的性能，还能够根据具体任务需求进行评估。因此，建议大家在正式使用该模型之前，先通过这种方式进行测试，以确保模型的能力和特性符合自身的需求。在测试过程中，用户可以尝试不同类型的输入和任务，从而更好地评估模型在实际应用中的表现。这一过程有助于发现模型的优势与局限，使用户在后续的应用中做出更为明智的选择。通过线上体验，还可以及时获取最新的功能更新和使用技巧，从而优化工作流程。

[Qwen2.5-72B-Instruct Hugging Face Space](https://huggingface.co/spaces/Qwen/Qwen2.5-72B-Instruct)

## 2. ModelScope本地部署流程

  - **Step 1. 创建conda虚拟环境**

Conda创建虚拟环境的意义在于提供了一个隔离的、独立的环境，用于Python项目和其依赖包的管理。每个虚拟环境都有自己的Python运行时和一组库。这意味着我们可以在不同的环境中安装不同版本的库而互不影响。根据官方文档信息建议Python版本3.10以上。创建虚拟环境的办法可以通过使用以下命令创建：

```bash
# python=3.11 指定了要安装的Python版本。你可以根据需要选择不同的名称和/或Python版本。

conda create -n qwen2_5 python=3.11
```

创建虚拟环境后，需要激活它。使用以下命令来激活刚刚创建的环境。如果成功激活，可以看到在命令行的最前方的括号中，就标识了当前的虚拟环境。虚拟环境创建完成后接下来安装torch。

如果忘记或者想要管理自己建立的虚拟环境，可以通过`conda env list`命令来查看所有已创建的环境名称。

如果需要卸载指定的虚拟环境则通过以下指令实现：

```
conda env remove --name envname
```

  - 需要注意的是无法卸载当前激活的环境，建议卸载时先切换到base环境中再执行操作。

  - **Step 2. 查看当前驱动最高支持的CUDA版本**

我们需要根据CUDA版本选择Pytorch框架，先看下当前的CUDA版本：

```
nvidia -smi
```

  - **Step 3. 在虚拟环境中安装Pytorch**

当前的电脑CUDA的最高版本要求是12.2，所以需要找到不大于12.2版本的Pytorch。

直接复制对应的命令，进入终端执行即可。这实际上安装的是为 CUDA 12.1 优化的 PyTorch 版本。这个 PyTorch 版本预编译并打包了与 CUDA 12.1 版本相对应的二进制文件和库。

  - **Step 4. 安装Pytorch验证**

待安装完成后，如果想要检查是否成功安装了GPU版本的PyTorch，可以通过几个简单的步骤在Python环境中进行验证：

```bash
import torch

print(torch.__version__)
```

如果输出是版本号数字，则表示GPU版本的PyTorch已经安装成功并且可以使用CUDA，如果显示ModelNotFoundError，则表明没有安装GPU版本的PyTorch，或者CUDA环境没有正确配置，此时根据教程，重新检查自己的执行过程。

当然通过pip show的方式可以很简洁的查看已安装包的详细信息。pip show \<package\_name\> 可以展示出包的版本、依赖关系（展示一个包依赖哪些其他包）、定位包安装位置、验证安装确实包是否正确安装及详情。

  - **Step 5. 安装必要的依赖包**

Transfomers是大模型推理时所需要使用的框架，官方给出的建议版本是`Transfomers>=4.37.0`，通过以下指令可以下载最新版本的Transfomers：

pip install transformers -U

安装完成后可以通过以下命令检查是否安装：

```
pip show transformers
```

接下来需要安装下载工具modelscope以及接下来要下载脚本的依赖accelerate，通过以下代码进行对应工具的部署：

```
pip install modelscope
pip install accelerate>=0.26.0
```

  - **Step 6.1 使用下载脚本安装**

> 这是一个自动安装并进行运行测试的脚本。

通过mkdir命令创建一个存放Qwen2.5项目文件的文件夹

```
mkdir qwen2_5 #文件的具体名称可以自定义
cd qwen2_5
```

在命令行种可以通过vim的方式创建或编辑文件，通过`vim download.py`创建一个python文件，将以下代码复制，然后保存退出。

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
 model_name,
 torch_dtype="auto",
 device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
 {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
 {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
 messages,
 tokenize=False,
 add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
 **model_inputs,
 max_new_tokens=512
)
generated_ids = [
 output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

编辑好脚本之后通过`python download.py`开始执行这个文件。

文件安装完毕后会启动对话测试（再次运行该文件会直接加载本地权重文件后直接进行运行推理），出现文本返回信息说明文件下载完整且可以启动。

文件完整性检查命令：

```python
pip install md5
```

```python
import hashlib

def calculate_md5(file_path):
    with open(file_path, 'rb') as f:
        md5_hash = hashlib.md5()
        while chunk := f.read(8192):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

# Example usage:
# file_path = 'your_file.pt'  # Replace with your actual file path
# md5_checksum = calculate_md5(file_path)
# print(f"MD5 checksum of {file_path}: {md5_checksum}")
```

## 三、ModelScope SDK部署流程

ModelScope 也提供了 RESTful API 或 SDK 供其在自己的应用中集成模型推理能力。

### 3.1 基本安装部署示例

下面是一个使用 ModelScope SDK 调用 Qwen2.5 模型的基本 Python 示例代码：

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Initialize the pipeline
pipe = pipeline(task=Tasks.text_generation, model='qwen/Qwen2.5-7B-Chat')

# Define the input
input_text = "你好，请简单介绍一下你自己。"

# Predict
result = pipe(input_text)

# Print the generated text
print(result)
```

通过以上步骤，用户可以便捷地在 ModelScope 平台上体验和部署 Qwen2.5 模型，从而将其强大的能力应用于各种智能应用场景中。

### 3.2 详细安装部署和测试示例

#### 前提条件

  * **Python环境：** 确保您已经安装了Python 3.8或更高版本。
  * **ModelScope库：** 需要安装ModelScope Python库。

#### 安装 ModelScope 库

如果您尚未安装ModelScope库，可以通过pip进行安装。建议在一个独立的Python虚拟环境中进行安装（例如使用`conda`或`venv`）。

```bash
# 建议创建并激活虚拟环境
# conda create -n modelscope_env python=3.9
# conda activate modelscope_env

# 安装modelscope库
pip install modelscope -U

# 如果需要支持特定的模型类型（如LLM），可能还需要安装额外的依赖，
# 但对于Qwen系列，通常modelscope本身会处理，或者在首次运行时提示。
# 对于LLM推理，确保transformers库也安装了最新版本
pip install transformers accelerate -U
```

#### 编写部署代码

您可以通过以下Python代码来加载并运行 `Qwen2.5-1.5B-Instruct` 模型进行推理。

```python
# 文件名: deploy_qwen2_5_sdk.py

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import json # 用于美观地打印JSON输出

# 1. 定义模型ID
# 这是Qwen2.5-1.5B-Instruct在ModelScope上的模型ID
# 确保您使用的ID是正确的，如果模型是私有的，您可能需要配置ModelScope token
model_id = 'qwen/Qwen2.5-1.5B-Instruct'

print(f"正在初始化ModelScope pipeline，模型ID: {model_id}...")

# 2. 初始化文本生成pipeline
# tasks.text_generation 适用于LLM的文本生成任务
# device='gpu' 或 'cpu' - 如果您有GPU，建议使用'gpu'以获得更快的推理速度。
# ModelScope SDK会自动在云端选择合适的计算资源。
text_generator = pipeline(
    task=Tasks.text_generation,
    model=model_id,
    device='gpu' # 如果本地不需要运行，这会影响云端资源的选择策略
                  # 对于SDK部署，ModelScope平台会在云端为您分配资源
                  # 这里的device参数通常指ModelScope SDK客户端在处理数据时的行为，
                  # 或者在某些本地模式下的行为。对于纯云端推理，此参数可能不直接控制云端设备。
)

print("Pipeline 初始化完成。开始进行推理...")

# 3. 准备输入：按照Qwen的chat_template格式组织对话
# Qwen系列Instruct模型通常遵循特定的对话格式（role: system, user, assistant）
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请简单介绍一下你自己。"}
]

# 将对话历史转换为模型所需的输入格式 (ModelScope pipeline通常会自动处理)
# 如果模型需要特定的提示词格式，可以在这里手动构建，但pipeline通常会帮你封装。
# 对于Qwen Instruct模型，通常直接传入字典列表即可。
input_text = {
    'text': messages
}

# 4. 执行推理
# result = text_generator(input_text)

# 实际调用时，ModelScope的pipeline对于chat模型可以直接传入messages
# 或者在某些版本中，input_text可以直接是字符串或messages列表
# 让我们尝试更通用的方法，直接传递用户query或messages
# 由于是Instruct模型，通常可以直接传入列表形式的对话历史

print("\n--- 第一次推理 ---")
print(f"用户输入: {messages[1]['content']}")
output = text_generator(messages) # 直接传递messages列表
print(f"模型输出:\n{json.dumps(output, indent=2, ensure_ascii=False)}")

# 从输出中提取生成的文本
generated_text_1 = output[0]['text'] if isinstance(output, list) and output else "未能获取输出"
print(f"提取的生成文本: {generated_text_1}")


# 5. 进行多轮对话 (示例)
print("\n--- 第二次推理 (多轮对话) ---")
messages.append({"role": "assistant", "content": generated_text_1}) # 将模型的回复加入历史
messages.append({"role": "user", "content": "那么，你对未来的AI发展有什么看法？"})

print(f"用户输入: {messages[-1]['content']}")
output_multi = text_generator(messages)
print(f"模型输出:\n{json.dumps(output_multi, indent=2, ensure_ascii=False)}")
generated_text_2 = output_multi[0]['text'] if isinstance(output_multi, list) and output_multi else "未能获取输出"
print(f"提取的生成文本: {generated_text_2}")


# 更多推理参数 (可选)
# 你可以在初始化pipeline时或调用时传入更多的generation_args
# 例如: max_length, top_p, temperature, do_sample 等

# text_generator = pipeline(
#     task=Tasks.text_generation,
#     model=model_id,
#     model_kwargs={'device_map': 'auto'} # 如果是在本地执行，并且模型支持的话
# )
# output = text_generator(messages, max_length=1024, temperature=0.7, top_p=0.9)

```

#### 执行代码

将上述代码保存为 `deploy_qwen2_5_sdk.py` 文件，然后在您的命令行或集成开发环境中运行：

```bash
python deploy_qwen2_5_sdk.py
```

#### 注意事项

  * **网络连接：** ModelScope SDK 部署依赖于网络连接，因为模型和推理都是在云端进行的。
  * **API 密钥/认证：** 对于公共模型，通常无需特殊的API密钥。但如果将来您部署或使用了私有模型，可能需要在ModelScope官网上获取API Token，并通过`modelscope.hub.snapshot_download`中的`token`参数或环境变量进行配置。
  * **计费：** ModelScope平台上的云端推理服务可能会涉及计费，具体费用请参考ModelScope的官方计费标准。对于小规模使用，通常有免费额度。
  * **模型版本：** 确保 `model_id` `qwen/Qwen2.5-1.5B-Instruct` 是ModelScope上可用的最新或您希望使用的特定版本。
  * **输出格式：** ModelScope `pipeline` 的输出格式可能因模型和任务而异，通常是一个包含字典的列表，字典中包含`'text'`键来表示生成的文本。

通过以上步骤，您就可以成功使用ModelScope SDK部署并与Qwen2.5-1.5B-Instruct模型进行交互了。

## 四、.Ollama框架部署流程

### 4.1 Ollama基本信息介绍

Ollama 是一个轻量级的、易于使用的工具，旨在简化在本地计算机上运行大型语言模型（LLMs）的过程。它提供了一个命令行界面（CLI），让用户可以方便地下载、运行和管理各种预训练的开源模型。Ollama 的主要特点是其极简主义设计，使得即使是没有深度机器学习背景的用户也能快速上手。

Ollama 的核心优势在于其便捷性。传统上，在本地部署 LLM 需要处理复杂的依赖关系、CUDA 配置以及模型权重管理。Ollama 将这些复杂性抽象化，用户只需几条简单的命令即可启动模型。它支持多种流行的模型架构，并且社区持续更新，不断有新的模型被集成进来。

此外，Ollama 也支持通过 Docker 进行部署，这进一步增强了其跨平台和环境的兼容性。Docker 容器化使得模型运行在一个隔离的环境中，避免了与系统其他组件的冲突，也方便了团队协作和生产环境的部署。

\<div align=center\>\<img src="[https://typora-photo1220.oss-cn-beijing.aliyuncs.com/LingYu/image-20241025152914107.png](https://www.google.com/search?q=https://typora-photo1220.oss-cn-beijing.aliyuncs.com/LingYu/image-20241025152914107.png)" width=80%\>\</div\>

### 4.2 使用Ollama实现Qwen2.5下载流程

在Linux环境下下载Ollama工具：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

下载完成后进行安装测试：

```bash
ollama run llama2 "Hello!"
```

可以通过以下方式检查Ollama的安装情况：

```bash
systemctl status ollama
```


```bash
ollama --version
```

下载Qwen2.5模型：

```bash
ollama pull qwen2.5
```

拉取完成后可以对Qwen2.5进行聊天测试：

```bash
ollama run qwen2.5
```

## 4.3 Ollama文件管理

```bash
ollama list
```

删除指定文件：

```bash
ollama rm qwen2.5
```

## 五、使用vLLM框架部署流程

## 5.1 vLLM基本介绍与安装

vLLM 是一个用于大型语言模型（LLM）推理的开源库，其设计目标是实现高吞吐量和低延迟。它通过一系列创新技术，如 PagedAttention（一种优化的注意力机制）和连续批处理，显著提升了 LLM 在 GPU 上的服务性能。

vLLM 的主要优势包括：

  - **高吞吐量**：通过高效的内存管理和请求调度，vLLM 能够同时处理更多的推理请求。
  - **低延迟**：通过减少不必要的计算和内存传输，vLLM 能够更快地生成响应。
  - **易于使用**：vLLM 提供了简单易用的 API，方便开发者集成到自己的应用中。
  - **广泛的模型支持**：vLLM 支持多种主流的 LLM 架构，包括 LLaMA、GPT-2、GPT-3 等。

**安装vLLM**

安装 vLLM 相对简单，可以通过 pip 进行安装。在安装之前，请确保您的系统满足以下要求：

  - **Python 环境**：Python 3.8 或更高版本。
  - **CUDA 环境**：vLLM 依赖 NVIDIA GPU 和 CUDA 工具包。请确保您的 CUDA 版本与 PyTorch 或其他深度学习框架兼容。
  - **PyTorch**：安装最新版本的 PyTorch。

您可以使用以下命令安装 vLLM：

```bash
pip install vllm
```

如果您需要支持特定的 CUDA 版本，可以参考 vLLM 的官方文档进行安装，通常会提供针对不同 CUDA 版本的轮子文件（wheel files）。

### 5.2 使用vLLM进行推理部署

可以通过直接在 Python 环境中调用 vLLM 进行推理。通过以下方式在 Python 中启动 vLLM 并指定所需的大模型，该方法更为简便高效。以下是对ModelScope本地部署、

## 六、Qwen大模型部署方式比较

以下是对ModelScope本地部署、ModelScope SDK部署、Ollama和vLLM不同部署方法的综合比较：

### 1. ModelScope本地部署

**特点：**
* **环境隔离：** 使用Conda创建独立的Python虚拟环境，便于管理项目依赖。
* **硬件要求：** 需要本地具备GPU硬件和相应的CUDA环境。
* **依赖管理：** 需手动安装PyTorch、Transformers、ModelScope、Accelerate等依赖库。
* **文件管理：** 模型文件和权重会下载到本地，方便离线使用和调试。
* **性能监控：** 可以直接监控本地GPU显存占用情况。

**优势：**
* 完全掌控本地环境和模型运行。
* 适合需要离线运行或对环境有特定要求的开发者。
* 便于深度定制和调试模型。

**劣势：**
* 环境配置相对复杂，依赖安装步骤较多。
* 对本地硬件（GPU显存）要求高。
* 不适合快速验证或资源受限的用户。

**典型场景：**
* 研究人员和开发者进行模型微调、实验或在本地进行高性能推理。
* 对数据隐私有严格要求的场景。

### 2. ModelScope SDK部署

**特点：**
* **云端服务：** 基于ModelScope平台提供的云端计算资源进行模型推理。
* **API调用：** 通过ModelScope SDK（Python）或RESTful API调用模型服务。
* **无需本地算力：** 用户无需购买和配置昂贵的GPU硬件。
* **环境简化：** 平台预置了模型运行所需的所有依赖和环境。
* **弹性伸缩：** 可根据需求动态调整计算资源。

**优势：**
* 部署过程简单快捷，无需处理复杂的本地环境配置。
* 无需本地GPU，降低硬件成本和门槛。
* 适合快速验证、功能迭代和团队协作。
* 服务稳定性高，支持高并发场景。

**劣势：**
* 依赖网络连接和ModelScope平台服务。
* 对于大规模、长时间的推理可能会产生平台费用。
* 对模型运行的底层细节控制较少。

**典型场景：**
* 个人开发者和中小型企业进行快速原型开发和应用部署。
* 需要按需付费、弹性伸缩的生产环境。
* 在线应用集成LLM能力。

### 3. Ollama

**特点：**
* **轻量级工具：** 专注于简化本地LLM的下载、运行和管理。
* **命令行界面（CLI）：** 提供简洁的命令进行模型操作。
* **易用性：** 即使没有深度机器学习背景也能快速上手。
* **多平台支持：** 提供Linux和Windows版本，也支持Docker部署。
* **模型生态：** 社区持续更新，集成多种流行的开源模型。

**优势：**
* 部署和管理LLM非常方便，大大降低了本地部署的复杂性。
* 适合快速体验不同开源LLM模型。
* 资源占用相对较低（取决于模型大小）。

**劣势：**
* 相比专业推理框架，可能在极限性能调优方面有所欠缺。
* 主要面向本地部署，不直接提供云服务能力。

**典型场景：**
* 个人用户、开发者在本地快速尝试和使用LLM。
* 教育和学习环境。
* 轻量级的本地应用集成。

### 4. vLLM框架部署

**特点：**
* **高性能推理：** 专为LLM推理设计，实现高吞吐量和低延迟。
* **优化技术：** 采用PagedAttention和连续批处理等创新技术提升GPU服务性能。
* **易于集成：** 提供简单易用的Python API。
* **广泛模型支持：** 支持多种主流LLM架构。
* **硬件要求：** 依赖NVIDIA GPU和CUDA工具包。

**优势：**
* 在LLM推理方面性能卓越，能显著提升效率。
* 适合对性能要求极高的生产环境。
* 能够最大化利用GPU资源。

**劣势：**
* 主要关注推理性能，对于模型训练和微调的直接支持较少。
* 同样需要本地GPU和CUDA环境，配置门槛相对高于Ollama。
* 对于资源受限或不需要极致性能的场景可能过于“重型”。

**典型场景：**
* 大规模在线推理服务。
* 对延迟和吞吐量有严格要求的AI应用。
* 模型部署到生产环境的专业解决方案。

### 综合比较总结

| 特性/方法      | ModelScope本地部署                               | ModelScope SDK部署                             | Ollama                                      | vLLM                                     |
| :------------- | :----------------------------------------------- | :--------------------------------------------- | :------------------------------------------ | :--------------------------------------- |
| **部署环境** | 本地（需GPU）                                    | 云端（ModelScope平台）                         | 本地（需GPU，或CPU，Docker）                | 本地（需NVIDIA GPU和CUDA）                 |
| **易用性** | 中等，需手动配置环境和依赖                     | 高，平台预置环境，API调用简单                  | 高，命令行工具，简化操作                  | 中等，需Python环境和依赖，但API简单       |
| **性能** | 取决于本地硬件和代码实现                       | 稳定，按需伸缩，性能由平台提供               | 一般，但已足够日常使用                      | 极高，专注于高吞吐和低延迟                |
| **资源消耗** | 占用本地GPU显存和存储                          | 无需本地硬件，按云端资源付费                 | 占用本地GPU显存和存储，相对轻量         | 占用本地GPU显存和存储，高效利用           |
| **控制力** | 高，完全控制模型和环境                         | 低，依赖平台服务                             | 中等，控制模型版本和运行，但底层细节隐藏 | 高，对推理过程有较强控制力               |
| **典型场景** | 模型研发、离线使用、深度定制                   | 快速开发、在线服务、资源受限用户             | 个人本地模型体验、学习、轻量级本地应用  | 大规模推理服务、对性能有极致要求场景     |

选择哪种部署方法取决于您的具体需求，包括可用的硬件资源、对易用性的偏好、对性能的要求、以及部署的场景（个人使用、开发测试还是生产环境）。

## 七、Qwen 2.5 部署后实验指南

本部分将引导学生进行一系列实验，以进一步熟悉和了解大型语言模型（LLM）的基本操作和特性，特别是在Qwen 2.5部署成功后。这些实验将涵盖基础文本生成、专业能力测试以及一些高级特性探索。

- 熟悉与部署好的Qwen 2.5模型进行交互。
- 了解Qwen 2.5模型在不同任务（如代码生成、数学问题解决）上的表现。
- 探索模型的上下文理解能力和指令遵循能力。
- 初步感知不同部署方式（如Ollama、vLLM）对模型性能的影响（可选）。

### 7.1 实验环境准备

在进行以下实验之前，请确保您已成功完成Qwen 2.5通用模型（例如`Qwen/Qwen2.5-7B-Instruct`）的部署（无论是ModelScope本地部署、ModelScope SDK部署、Ollama还是vLLM部署），并能正常启动模型进行推理。

### 7.2 实验内容

### 实验一：基础文本生成与对话

本实验将演示如何使用Python脚本通过ModelScope本地部署的Qwen 2.5通用模型进行基础文本生成和多轮对话。

**所需模型：** `Qwen/Qwen2.5-7B-Instruct` (或其他`Instruct`版本)

```python
# 文件名: experiment_1_basic_chat.py

from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 加载模型和分词器
# 确保你有足够的GPU显存。如果显存不足，可以尝试加载更小的模型（如3B）或使用量化版本。
# 如果没有GPU，或者显存严重不足，可以将device_map设为None，并手动将模型部分移动到CPU。
# model_name = "qwen/Qwen2.5-3B-Instruct" # 示例：更小的模型
model_name = "qwen/Qwen2.5-7B-Instruct" # 常用模型

print(f"正在加载模型: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto", # 自动选择精度 (fp16/bf16)
    device_map="auto"   # 自动将模型分配到可用的设备 (GPU/CPU)
)
model.eval() # 将模型设置为评估模式

print("模型加载完成。开始对话...")

# 2. 进行简单对话
print("\n--- 简单对话示例 ---")
prompt_1 = "你好，请简单介绍一下你自己。"
messages_1 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_1}
]
text_1 = tokenizer.apply_chat_template(
    messages_1,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs_1 = tokenizer([text_1], return_tensors="pt").to(model.device)

generated_ids_1 = model.generate(
    **model_inputs_1,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
response_1 = tokenizer.batch_decode(generated_ids_1[:, model_inputs_1.input_ids.shape[1]:], skip_special_tokens=True)[0]
print(f"用户: {prompt_1}")
print(f"模型: {response_1}")

# 3. 进行多轮对话
print("\n--- 多轮对话示例 ---")
messages_multi = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你觉得未来人工智能会如何发展？"}
]

for i in range(2): # 进行2轮追加对话
    text_multi = tokenizer.apply_chat_template(
        messages_multi,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs_multi = tokenizer([text_multi], return_tensors="pt").to(model.device)

    generated_ids_multi = model.generate(
        **model_inputs_multi,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    response_multi = tokenizer.batch_decode(generated_ids_multi[:, model_inputs_multi.input_ids.shape[1]:], skip_special_tokens=True)[0]

    print(f"用户: {messages_multi[-1]['content']}")
    print(f"模型: {response_multi}")

    # 将模型的回复添加到对话历史中，以便进行下一轮对话
    messages_multi.append({"role": "assistant", "content": response_multi})
    if i == 0:
        messages_multi.append({"role": "user", "content": "那么，它对教育会有什么影响？"})
    else:
        print("对话结束。")

```

**执行方法：**
将上述代码保存为 `experiment_1_basic_chat.py` 文件，然后在您的Conda虚拟环境中运行：

```bash
python experiment_1_basic_chat.py
```

### 实验二：上下文长度与指令遵循

本实验将测试Qwen 2.5通用模型处理长文本的能力，并观察其在多条件指令和角色扮演任务中的表现。

**所需模型：** `Qwen/Qwen2.5-7B-Instruct` (或其他`Instruct`版本)

```python
# 文件名: experiment_2_context_instruction.py

from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 加载模型和分词器
model_name = "qwen/Qwen2.5-7B-Instruct"

print(f"正在加载模型: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

print("模型加载完成。开始上下文和指令遵循实验...")

def generate_response(messages, max_tokens=512):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.im_end_id]
    )
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response

# 2. 长文本摘要
print("\n--- 长文本摘要 ---")
long_text = """
人工智能（AI）正在以前所未有的速度改变着世界。从自动驾驶汽车到智能医疗诊断，再到个性化推荐系统，AI技术已经渗透到我们生活的方方面面。最近，大型语言模型（LLM）的飞速发展更是将AI推向了新的高潮。这些模型，如GPT系列和Qwen系列，能够理解、生成和处理人类语言，展现出惊人的对话、写作和编程能力。然而，随着AI能力的增强，关于其伦理、安全和就业影响的讨论也日益增多。如何在推动技术进步的同时，确保AI的负责任发展，成为全球社会面临的重要课题。未来，AI有望在更多领域发挥关键作用，但我们也需警惕潜在的风险，并制定相应的政策和法规来引导其健康发展。教育、医疗、金融、艺术等行业都将因AI而发生深刻变革。例如，在教育领域，AI可以提供个性化学习路径，辅助教师进行教学；在医疗领域，AI可以加速新药研发，辅助医生进行疾病诊断。这些进步无疑将极大地提升人类的生活质量和工作效率。然而，我们也必须认识到，AI并非万能，它仍然存在局限性，例如对复杂常识的理解不足、可能产生偏见等。因此，人与AI的协作，而不是AI完全取代人类，将是未来的主要模式。
"""
messages_summary = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"请总结以下文章的核心内容：\n{long_text}"}
]
response_summary = generate_response(messages_summary, max_tokens=200) # 限制摘要长度
print(f"用户（摘要请求）:\n{messages_summary[-1]['content'][:50]}...") # 打印部分原文
print(f"模型摘要:\n{response_summary}")

# 3. 多条件指令遵循
print("\n--- 多条件指令遵循 ---")
complex_instruction = "请写一篇关于未来交通的短文，字数在150字以内，并包含至少两个创新技术（如飞行汽车、超高速列车），同时语气要积极向上，不要出现否定词。"
messages_complex = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": complex_instruction}
]
response_complex = generate_response(messages_complex, max_tokens=250) # 适当放宽生成上限以观察
print(f"\n用户: {complex_instruction}")
print(f"模型回复:\n{response_complex}")

# 4. 角色扮演
print("\n--- 角色扮演 ---")
messages_roleplay = [
    {"role": "system", "content": "你现在是一位经验丰富的历史老师，请用这个身份向我介绍秦朝的历史。"},
    {"role": "user", "content": "老师，请您介绍一下秦朝的历史吧。"}
]
response_roleplay = generate_response(messages_roleplay)
print(f"\n用户: 老师，请您介绍一下秦朝的历史吧。")
print(f"模型（历史老师）:\n{response_roleplay}")

```

**执行方法：**
将上述代码保存为 `experiment_2_context_instruction.py` 文件，然后在您的Conda虚拟环境中运行：

```bash
python experiment_2_context_instruction.py
```

### 实验三（高级/可选）：不同部署方式的性能比较

本实验将简要演示如何比较Ollama和vLLM在推理速度和资源占用上的差异。这需要您同时部署Ollama和vLLM。

**所需模型：** `Qwen/Qwen2.5-7B-Instruct` (或其Ollama兼容版本)

#### 3.1 Ollama 性能测试 (命令行)

1.  **启动Qwen 2.5模型 (Ollama):**
    在终端中运行：
    ```bash
    ollama run qwen2.5 # 如果没有qwen2.5模型，请先执行 ollama pull qwen2.5
    ```
2.  **进行交互并计时：**
    当模型启动后，手动输入一个长一点的提示，并观察模型开始生成到完成的时间。
    例如：
    `请用大约500字的篇幅，详细阐述大型语言模型（LLM）的最新发展趋势、面临的挑战以及未来可能的应用方向。`
    在模型生成过程中，您可以在另一个终端运行 `nvidia-smi` 命令来观察GPU显存占用情况。

#### 3.2 vLLM 性能测试 (Python脚本)

为了更精确地测量时间，我们将使用Python的`time`模块。

```python
# 文件名: experiment_3_vllm_performance.py

from vllm import LLM, SamplingParams
import time
import torch # 用于检查CUDA状态

# 1. 检查CUDA可用性 (vLLM需要GPU)
if not torch.cuda.is_available():
    print("CUDA 不可用，vLLM需要GPU才能运行。请确保您的系统有NVIDIA GPU并正确配置了CUDA。")
    exit()

# 2. 加载LLM (Qwen 2.5)
# 注意：vLLM加载模型的方式与ModelScope有所不同，它通常直接从Hugging Face模型ID加载
# 请确保您已通过vLLM支持的方式安装了Qwen 2.5模型或其Hugging Face路径可访问。
# 通常，vLLM会在首次运行时自动下载模型。
model_name_vllm = "Qwen/Qwen2.5-7B-Instruct"

print(f"正在加载vLLM模型: {model_name_vllm}...")
llm = LLM(model=model_name_vllm)
print("vLLM模型加载完成。")

# 3. 定义采样参数和测试提示
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=500)
test_prompt = "请用大约500字的篇幅，详细阐述大型语言模型（LLM）的最新发展趋势、面临的挑战以及未来可能的应用方向。"

# 4. 执行推理并计时
print("\n--- vLLM 推理性能测试 ---")
start_time = time.time()
outputs = llm.generate([test_prompt], sampling_params)
end_time = time.time()

# 5. 打印结果和时间
generated_text = outputs[0].outputs[0].text
print(f"用户: {test_prompt}")
print(f"vLLM 生成文本:\n{generated_text}")
print(f"\n推理耗时: {end_time - start_time:.2f} 秒")

# 6. 观察GPU显存占用
# 可以在模型加载和推理过程中，在另一个终端运行 `nvidia-smi` 命令来观察GPU显存占用情况。
# vLLM通常会预分配较多的显存以优化性能。

```

**执行方法：**
将上述代码保存为 `experiment_3_vllm_performance.py` 文件，然后在您的Conda虚拟环境中运行：

```bash
python experiment_3_vllm_performance.py
```

**性能比较与讨论：**
在执行完Ollama和vLLM的测试后，对比两者生成相同文本（或大致相同长度文本）所花费的时间和观察到的GPU显存占用。

  * **推理速度：** 通常情况下，vLLM由于其优化的Attention机制和批处理能力，在处理多个并发请求或长文本生成时，可能会比Ollama更快。对于单次短文本生成，差异可能不那么明显。
  * **资源占用：** vLLM为了追求高性能，通常会预分配更多的GPU显存。Ollama在某些情况下可能显得更为“轻量”。

通过这些实验，学生将更直观地理解不同部署方案的实际效果和性能特点。

## Qwen 2.5大模型部署课程总结

本课程旨在为学员提供一个全面而实用的Qwen 2.5系列大型语言模型（LLM）的部署与应用指南。从模型的基础认知到多种部署策略的实践，本课程致力于帮助学员掌握在不同场景下高效利用Qwen 2.5模型的能力。

### 课程目标回顾

通过本课程的学习，您已能够：
* **理解Qwen 2.5模型概况：** 掌握Qwen 2.5系列模型的最新特性、不同参数规模版本及其在通用任务上的性能表现。
* **掌握多种部署方法：** 深入了解并能够实践ModelScope本地部署、ModelScope SDK云端部署、Ollama轻量级部署以及vLLM高性能推理框架部署。
* **进行模型基础操作与特性探索：** 熟悉与部署好的Qwen 2.5模型进行交互，包括基础文本生成、多轮对话、长文本摘要、复杂指令遵循和角色扮演等。
* **初步感知性能差异：** 对比不同部署方式（如Ollama与vLLM）在推理速度和资源占用上的特点，为实际应用场景选择提供依据。

### 核心内容精炼

1.  **Qwen 2.5模型介绍：** 课程首先详细介绍了阿里巴巴通义千问团队开发的Qwen 2.5系列模型，包括其作为稠密、decoder-only语言模型的特性，以及0.5B到72B的多种参数规模。我们还提及了其专家模型Qwen2.5-Coder和Qwen2.5-Math，尽管在实践部分未深入，但为学员建立了全面的模型认知。

2.  **部署方法深度对比：** 课程详细对比了四种主流部署方案：
    * **ModelScope本地部署：** 强调其环境掌控性高、适合离线和深度定制的优势，但对本地硬件和配置要求较高。
    * **ModelScope SDK部署：** 突出其云端服务、无需本地算力、部署快捷和易于集成的特点，是快速验证和云端集成的理想选择。
    * **Ollama：** 作为一个轻量级工具，它极大简化了本地LLM的下载、运行和管理，降低了本地体验门槛。
    * **vLLM：** 作为高性能推理框架，其通过PagedAttention和连续批处理等技术，实现了LLM的高吞吐量和低延迟，适用于对性能有严苛要求的生产环境。

3.  **ModelScope SDK部署实践：** 课程提供了使用ModelScope SDK部署Qwen2.5-1.5B-Instruct的详细步骤和可执行代码，让学员能够直接体验云端模型调用的便捷性。

4.  **模型交互与特性实验：** 通过一系列精心设计的实验，学员亲自动手：
    * 使用Python脚本进行Qwen 2.5的基础文本生成和多轮对话。
    * 探索模型在长文本摘要、多条件指令遵循以及角色扮演任务中的表现，加深对模型理解能力和指令遵循能力的认识。
    * （可选）通过Ollama命令行和vLLM Python脚本的对比，直观感受不同推理框架在性能上的差异。

### 获得的实践技能

完成本课程后，您将具备：
* 独立在本地或利用云平台部署Qwen 2.5通用大模型的能力。
* 使用Python编程语言与大模型进行交互，并实现文本生成、对话、摘要等功能。
* 根据实际需求（如硬件资源、性能要求、易用性）选择合适的LLM部署方案的判断力。
* 通过实际操作，对大模型的上下文理解、指令遵循等核心特性有更直观的理解。

### 展望与建议

Qwen 2.5系列模型作为国产大模型的重要代表，在各项能力上持续进步。掌握其部署与应用，是您进入AI大模型时代的关键一步。未来，随着模型技术和部署工具的不断发展，模型部署将更加便捷高效。建议您继续关注Qwen及其他主流大模型的最新进展，探索更多高级应用场景，如模型微调、 Agent开发、以及与RAG（检索增强生成）等技术的结合，以充分发挥大模型的潜力。