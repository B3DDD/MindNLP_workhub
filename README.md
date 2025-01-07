# MindNLP学习营

MindNLP第五期学习营作业

简述：文件夹按每节课进行分类，每节课会上传标注好的课件以及完成跑通的作业代码

1st
第一课作业.txt
https://gitee.com/mindspore/community/issues/IB7HY9 


2nd

课件：
《极简风的大模型微调实战-课件(标注版）》

原课件知识的基础上，利用大模型AI助手逐一对专业名词进行释义，并且利用大模型理解图示内容进行批注。

代码：
《IA3_sequence_classification.ipynb》根据mindnlp的ia3仓库中的标准答案《sequence_classification.ipynb》略加修改而来。
原标准答案代码有一点问题：所有训练结果都会显示准确率为68%。
通过提issue已经把问题解决，代码可以正常更新模型参数。
该现象成因已经探明，写在CSDN博文中：【昇思打卡营第五期(MindNLP特辑）第二课--RoBERTa-Large的IA3微调 - CSDN App】
https://blog.csdn.net/a1966565/article/details/144751602?sharetype=blog&shareId=144751602&sharerefer=APP&sharesource=a1966565&sharefrom=link

-adapter_config.json和adapter_model.ckpt是该代码运行后保存的模型参数，可以通过代码重新加载使用。

3rd