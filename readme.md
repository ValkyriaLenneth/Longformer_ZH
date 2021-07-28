# 中文预训练Longformer模型 | Longformer_ZH with PyTorch 

相比于Transformer的O(n^2)复杂度，Longformer提供了一种以线性复杂度处理最长4K字符级别文档序列的方法。Longformer Attention包括了标准的自注意力与全局注意力机制，方便模型更好地学习超长序列的信息。

Compared with O(n^2) complexity for Transformer model, Longformer provides an efficient method for processing long-document level sequence in Linear complexity. Longformer’s attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. 

我们注意到关于中文Longformer或超长序列任务的资源较少，因此在此开源了我们预训练的中文Longformer模型参数， 并提供了相应的加载方法，以及预训练脚本。

There are not so much resource for Chinese Longformer or long-sequence-level chinese task. Thus we open source our pretrained longformer model to help the researchers.
## 加载模型 | Load the model
您可以使用谷歌云盘或百度网盘下载我们的模型  
You could get Longformer_zh from Google Drive or Baidu Yun.

- Google Drive: https://drive.google.com/file/d/1h0oh6hmjc0w3n21VburjiZPJbChRSS4n/view?usp=sharing
- 百度云:  链接：https://pan.baidu.com/s/1tgAOd7SuWxbwTRSagN0lyg 提取码：bdgb

我们同样提供了Huggingface的自动下载  
We also provide auto load with HuggingFace.Transformers.
```
from Longformer_zh import LongformerZhForMaksedLM
LongformerZhForMaksedLM.from_pretrained('ValkyriaLenneth/longformer_zh')
```

## 注意事项 | Notice
- 区别于英文原版Longformer， 中文Longformer的基础是Roberta_zh模型，其本质上属于 `Transformers.BertModel` 而非 `RobertaModel`, 因此无法使用原版代码直接加载。
- Different with origin English Longformer, Longformer_Zh is based on Roberta_zh which is a subclass of `Transformers.BertModel` not `RobertaModel`. Thus it is impossible to load it with origin code.
- 我们提供了修改后的中文Longformer文件，您可以使用其加载参数。
- We provide modified Longformer_zh class, you can use it directly to load the model. 
- 如果您想将此参数用于更多任务，请参考`Longformer_zh.py`替换Attention Layer.
- If you want to use our model on more down-stream tasks, please refer to `Longformer_zh.py` and replace Attention layer with Longformer Attention layer.

## 关于预训练 | About Pretraining
- 我们的预训练语料来自 https://github.com/brightmart/nlp_chinese_corpus， 根据Longformer原文的设置，采用了多种语料混合的预训练数据。
- The corpus of pretraining is from https://github.com/brightmart/nlp_chinese_corpus. Based on the paper of Longformer, we use a mixture of 4 different chinese corpus for pretraining.
- 我们的模型是基于Roberta_zh_mid (https://github.com/brightmart/roberta_zh),训练脚本参考了https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb
- The basement of our model is Roberta_zh_mid (https://github.com/brightmart/roberta_zh). Pretraining scripts is modified from https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb.

- 同时我们在原版基础上，引入了 `Whole-Word-Masking` 机制，以便更好地适应中文特性。
- We introduce `Whole-Word-Masking` method into pretraining for better fitting Chinese language.
- `Whole-Word-Masking`代码改写自TensorFlow版本的Roberta_zh，据我们所知是第一个开源的Pytorch版本WWM.
- Our WWM scripts is refacted from Roberta_zh_Tensorflow, as far as we know, it is the first open source Whole-word-masking scripts in Pytorch.

- 模型 `max_seq_length = 4096`, 在 4 * Titan RTX 上预训练3K steps 大概用时4天。
- Max seuence length is 4096 and the pretraining took 4 days on 4 * Titan RTX.
- 我们使用了 `Nvidia.Apex` 引入了混合精度训练，以加速预训练。
- We use `Nvidia.Apex` to accelerate pretraining.
- 关于数据预处理， 我们采用 `Jieba` 分词与`JIONLP`进行数据清洗。
- We use `Jieba` Chinese tokenizer and `JIONLP` data cleaning.
- 更多细节可以参考我们的预训练脚本
- For more details, please check our pretraining scripts.

## 更新计划 | Update Plan
- 我们首先会放出预训练3K-steps的模型
- We released our 3K-steps pretrained model.
- 在八月将开源训练15K-steps的模型
- We will release our 15K-steps full pretrained model in August.

## 效果测试 | Evaluation
### CCF Sentiment Analysis
- 由于中文超长文本级别任务稀缺，我们仅采用CCF-Sentiment-Analysis任务进行测试
- Since it is hard to acquire open-sourced long sequence level chinese NLP task, we only use CCF-Sentiment-Analysis for evaluation. 

|Model|Dev F|
|----|----|
|Bert|80.3|
|Bert-wwm-ext| 80.5|
|Roberta-mid|80.5|
|Roberta-large|81.25|
|Longformer_SC|79.37|
|Longformer_ZH|80.51|

### Pretraining BPC
- 我们提供了预训练BPC(bits-per-character), BPC越小，代表语言模型性能更优。可视作PPL.
- We also provide BPC scores of pretraining, the lower BPC score, the better performance Langugage Model has. You can also treat it as PPL.

|Model|BPC|
|---|---|
|Longformer before training| 14.78|
|Longformer after training| 3.10|

## 致谢
感谢东京工业大学 奥村·船越研究室 提供算力。

Thanks Okumula·Funakoshi Lab from Tokyo Institute of Technology who provides the devices and oppotunity for me to finish this project.


