# MyCNN</br>
2019/07/11 初次上传</br>
卷积和全连接神经网络结构由参数控制；</br>
运行于jupyter notebook，还在完善中，暂不转为.py文件，但主要目标已实现，目前正在完善细节；</br>
实验数据为mnist数据集；</br>
函数及函数调用方法见DNN_Essay.ipynb，一目了然；</br>
已知问题：</br>
1、macOS Mojave 10.14.5下的jupyter notebook运行此程序会崩溃，未知其他型号、其他系统版本的苹果设备
是否有此问题，该设备下pd虚拟Win10没有此问题；</br>
2、mnist数据集不知从何时起，每个图片标签变为对应数字，而非10维向量，故示例中对标签做了处理。</br>
（已于2019/07/15解决：读取mnist数据时，没有加入one_hot=True）</br>

更新记录</br>
2019/07/12 加入模型固化和读取功能</br>
2019/07/18 将mnist数据集存入TFRecord(见TFRecord_file.ipynb)</br>

