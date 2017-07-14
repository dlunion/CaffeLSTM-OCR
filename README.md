# CaffeLSTM-OCR
基于caffe的LSTM CTC OCR案例，能够利用该案例完成序列的识别，包括验证码、车牌、身份证号码、地址等长序列动长的内容识别<br/>
这是一个resnet+blstm的例子，blstm是双向lstm的意思，resnet也只是采用了其中的126部分，丢掉了一大半<br/>

这个最大的贡献，是<br/>
能够训练长序列的ocr识别，可以使用这个技术完成比如身份证号码、地址、车牌等识别任务<br/>


lstm网络设计注意事项：<br/>
1.保证CNN得到的featuremap输入到lstm时的宽度至少大于等于最大字符数的3倍左右，即time_step大于等于最大字符数3倍，否则小了不行<br/>
2.如果是配合完整的resnet精度应该能够更好<br/>
3.这次训练的精度为100%停止的，测试精度是97%左右，算是对复杂验证码OCR的一个证明，证明能力<br/>
4.对于自己衔接网络，只要保证最后的time_step能配的上就不会有错<br/>
5.训练过程中，如果出现难以收敛，把dropout层的dropout_ratio调低到0.5或者更低比如0.3甚至0，如果过拟合了，就调高，甚至可以0.7、0.9。当然默认是不要修改他，除非你也在研究<br/>
6.lstm的num_output个数也影响精度，还有所谓的多层lstm也是可以有的<br/>


<br/>
里面的C++程序是纯无依赖的，只依赖[CCDL](https://github.com/dlunion/CCDL)
<br/>

# 下载
模型、演示图片、和依赖项<br/>
[CaffeLSTM-OCR.rar](http://www.zifuture.com/fs/12.github/CaffeLSTM-OCR/CaffeLSTM-OCR.rar)