# Model-ONNX

### bert model prompt

采用 pi coding agent 输入以下prompt，即可获得 build_bert.py;
运行 python build_bert.py 可得到 bert.onnx 模型；
最后通过 netron app 打开得到模型拓扑。

```bash
帮我采用onnx的api手动构建一个bert模型，并输出onnx模型，名字叫bert.onnx，不需要创建一个整图，通过netron打开可视化图后，可以看到的是12层bert encoder layer，再打开这一层可以看到层内具体的算子和连接关系
```

