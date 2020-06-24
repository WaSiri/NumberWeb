# NumberWeb
## use mnist model and flask to set up a web that could predict number
### 项目说明
* Flask:项目文件
*     Models：内含四个文件，为tensorflow基于MNIST数据集训练所得模型文件
*     Static：内含两个文件，为网页index界面的css与js文件
*     Templates：内含两个文件，base.html为模板页，index.html为子页，效果如下图所示。
*     app.py：整个项目的启动文件
*     test.py：训练模型的启动文件，输入图片，返回识别结果
*     util.py：处理json文件转为img
* img:测试图片

### 使用说明
* 环境配置
* 第一次使用时右键app.py文件，点击Run ” Flask(app.py)” 
 ，之后使用该项目便可以简单使用右上角的运行
* 找到运行结果，点击本地网址 http://127.0.0.1:5000/
 
* 从img文件中导入web界面一张图片，图片框以及下方均会显示缩略图
 
* 点击识别，图片框内仍是缩略图，下方变为预测结果


