# MindNLP学习营

## MindNLP第五期学习营作业

简述：文件夹按每节课进行分类，每节课会上传标注好的课件以及完成跑通的作业代码

---------------------------------------------------------------------------------------------------------------------------------------------------------
# 1st


第一课作业.txt
https://gitee.com/mindspore/community/issues/IB7HY9 


-----------------------------------------------------------------------------------------------------------------------------------------------------------
# 2nd

## 课件：
《极简风的大模型微调实战-课件(标注版）》

原课件知识的基础上，利用大模型AI助手逐一对专业名词进行释义，并且利用大模型理解图示内容进行批注。

## 代码：
《IA3_sequence_classification.ipynb》根据mindnlp的ia3仓库中的标准答案《sequence_classification.ipynb》略加修改而来。
原标准答案代码有一点问题：所有训练结果都会显示准确率为68%。
通过提issue已经把问题解决，代码可以正常更新模型参数。
该现象成因已经探明，写在CSDN博文中：

【昇思打卡营第五期(MindNLP特辑）第二课--RoBERTa-Large的IA3微调 - CSDN App】
https://blog.csdn.net/a1966565/article/details/144751602?sharetype=blog&shareId=144751602&sharerefer=APP&sharesource=a1966565&sharefrom=link

-adapter_config.json和adapter_model.ckpt是该代码运行后保存的模型参数，可以通过代码重新加载使用。



------------------------------------------------------------------------------------------------------------------------------------------------------
# 3rd

## 《数据并行（标注版）》

原课件知识的基础上，利用大模型AI助手逐一对专业名词进行释义，并且利用大模型理解图示内容进行注释。

## 代码

https://github.com/mindspore-lab/mindnlp/tree/master/examples/parallel/bert_imdb_finetune

## 训练过程及产生的诸多信息和文件，专门写了一篇博文来解释：

《昇思打卡营第五期(MindNLP特辑）第三课--基于MindNLP的数据并行训练-上：课程示例代码详解》
[https://blog.csdn.net/a1966565/article/details/144991037?spm=1001.2014.3001.5501](https://blog.csdn.net/a1966565/article/details/144991037?fromshare=blogdetail&sharetype=blogdetail&sharerId=144991037&sharerefer=PC&sharesource=a1966565&sharefrom=from_link)

上传文件说明：

![image](https://github.com/user-attachments/assets/8fa1582b-c545-460f-ab7e-de324e7e0bba)

完整文件如图。

由于optimizer.ckpt和model .safetensors文件太大而无法上传。

-----------------------------------------------------------------------------------------------------------------------------------------------------
# 4th


## 《大模型Decoding+MindSpore+NLP分布式推理详解》标注版

原课件基础上，利用AI助手对于专业名词、图表和公式进行注释。方便需要更详细解释的同学们想学习。

## generate_parameter.ipynb

课程提到的第一篇代码，展示了多种Decoding策略的代码和计算原理，并给出了一定的反馈值。

## run_llama3.py

Llama3模型的非并行代码，尽量在足够大的显存环境下运行。

## distributed文件夹内

《昇思打卡营第五期(MindNLP特辑）课程四：基于MindSpore NLP的LLM推理（decoding策略）-上：课程示例代码详解》

https://blog.csdn.net/a1966565/article/details/145110468?fromshare=blogdetail&sharetype=blogdetail&sharerId=145110468&sharerefer=PC&sharesource=a1966565&sharefrom=from_link

博文的主要内容在这个文件夹里

根据多种解码策略改动的llama3模型分布式运行代码，需要在msrun或者mpirun的方式下启动。

msrun is a MindSpore defined launcher for multi-process parallel execution, which can get best performance, you can use it by the command below:

msrun --worker_num=2 --local_worker_num=2 --master_port=8118 --join=True run_llama3_distributed.py
if you use Ascend NPU with Kunpeng CPU, you should bind-core to get better performance

msrun --worker_num=2 --local_worker_num=2 --master_port=8118 --join=True --bind_core=True run_llama3_distributed.py


mpirun controls several aspects of program execution in Open MPI, you can use it by the command below:

mpirun -n 2 python run_llama3_distributed.py
if you use Ascend NPU with Kunpeng CPU, you should bind-core to get better performance:

mpirun --bind-to numa -n 2 python run_llama3_distributed.py

## enhanced文件夹内

《昇思打卡营第五期(MindNLP特辑）课程四：基于MindSpore NLP的LLM推理（decoding策略）-下：中文成语释义与解析挑战》

https://blog.csdn.net/a1966565/article/details/145119113?fromshare=blogdetail&sharetype=blogdetail&sharerId=145119113&sharerefer=PC&sharesource=a1966565&sharefrom=from_link

博文主要内容在这个文件夹

主要是中文成语释义与挑战的赛事代码baseline，在这个文件夹内，包含了使用QWEN2-7B-instruct、GLM4的API、transformer、vLLM来处理赛题的代码。

-------------------------------------------------------------------------------------------------------------------------------------
# 5th

## 





