# 平台知识库目录

平台智能调控助手的知识检索只读取本目录下的 UTF-8 文本文件（`.txt` / `.md`），可包含子目录。建议按来源组织：

```text
knowledge/
├─ theory/          电力系统理论教材与专业资料
├─ nerc/            NERC Lessons Learned 运行事件摘录
└─ domestic/        国内规程、市场规则与运行案例
```

要求：

- 只放有使用权的资料；原始语料、向量索引和评测数据不随源码分发。
- 不要把 PowerGridQA 评测 JSONL 放在本目录；它属于 `backend/data/powergridqa/`，仅用于评测。
- 知识片段应保留来源名称、章节或事件编号；平台会把相对路径作为 `source_id` 返回。

未放置语料时，助手仍可读取实时功率、气象、预测和 NWP 并给出规则兜底建议，但知识检索命中数为 0。

如需知识库或数据合作，请联系 [ts26170007a31@cumt.edu.cn](mailto:ts26170007a31@cumt.edu.cn)。
