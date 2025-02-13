# DPO (Preference Optimization) PyTorch 实现

DPO（Preference Optimization）是一种优化模型生成输出的方法，通过学习人类对话数据中，偏好输出与不偏好输出之间的概率差异，调整模型生成的输出，使得它们更加符合人类的偏好。

本实现基于 LLaMA（Large Language Model Meta AI）模型，利用偏好数据集与拒绝数据集来计算概率差异，并使用 `logsigmoid` 函数计算损失值，最后通过反向传播优化模型。

## 依赖项

以下是运行此代码所需的 Python 包：

- `torch`（PyTorch）
- `transformers`（Hugging Face Transformers）
- `numpy`

安装依赖项的命令：

```bash
pip install torch transformers numpy
```

## 使用方法

1. 克隆此仓库到本地：
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. 运行训练代码：
    ```bash
    python dpo.py
    ```

3. 查看优化过程和更新后的模型参数。

## 代码解释

### 1. 加载预训练模型

本代码使用 LLaMA 模型作为基础模型，使用 `LlamaConfig` 和 `LlamaForCausalLM` 来配置和加载模型。

```python
config = LlamaConfig(
    vocab_size=32,  # 词表的尺寸
    hidden_size=512,  # 隐藏层尺寸
    num_attention_heads=4,  # 注意力头数量
    num_key_value_heads=4  # 键值对数量
)
ref_model = LlamaForCausalLM(config)  # 加载预训练的模型
```

### 2. 数据准备

数据集包括偏好数据集（`prompt_chosen`）和拒绝数据集（`prompt_rejected`）。每个数据集都与相应的注意力掩码（`attention_mask`）一起使用。

- `prompt_chosen`：表示偏好数据（正向样本）。
- `prompt_rejected`：表示拒绝数据（负向样本）。
- `attention_mask`：用于标记模型应关注的部分。

### 3. 概率计算

使用 `get_probs` 函数计算每个 token 在词汇表中对应标签的概率。通过 `log_softmax` 得到每个 token 的 log 概率，并通过 `torch.gather` 获取相应标签的概率。

```python
def get_probs(logits, labels):
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return per_token_logps
```

### 4. DPO 损失函数

在训练过程中，损失值基于偏好数据集和拒绝数据集的概率差异进行计算。使用 `logsigmoid` 函数将 logits 转化为概率，并乘以标签进行损失值的计算。

```python
losses = -F.logsigmoid(beta * logits) * labels.float()  # 计算损失
```

### 5. 模型优化

通过反向传播优化模型参数：

```python
optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer.zero_grad()  # 清空梯度
loss.backward()  # 计算梯度
optimizer.step()  # 更新参数
```

## 优化与训练

1. **偏好数据集和拒绝数据集**：通过对比偏好数据集与拒绝数据集之间的概率差异来训练模型，优化模型生成输出的偏好。

2. **损失计算**：使用 `logsigmoid` 函数计算损失，其中标签（`labels`）决定了哪些部分的损失需要参与计算。

3. **反向传播**：通过 `loss.backward()` 和优化器 `optimizer.step()` 更新模型参数，使模型能够生成更符合人类喜好的输出。

## 模型与数据集

本实现使用 LLaMA 模型作为基础模型。数据集是人工生成的偏好与拒绝数据集，模拟了用户的喜好和拒绝。

### 数据集说明

- **偏好数据集（Chosen）**：表示用户偏好的样本，用于训练模型生成符合用户偏好的输出。
- **拒绝数据集（Rejected）**：表示用户不偏好的样本，用于训练模型避免生成不符合用户偏好的输出。

### 模型说明

LLaMA（Large Language Model Meta AI）是一个强大的预训练语言模型，适用于各种自然语言处理任务。本实现使用 LLaMA 模型来进行 DPO 训练，使其在推理过程中生成更加符合人类偏好的输出。

## 许可证

该项目基于 MIT 许可证开源。
