# 系统架构

平台采用前后端分离和数据适配层设计。Vue 前端通过 REST 与流式接口访问 FastAPI；后端将 MySQL 功率数据、气象观测、预测结果和本地 NWP 快照转换为统一场站领域模型；智能调控服务在此上下文上执行意图识别、知识检索、策略评估与回答生成。

```mermaid
flowchart LR
    DB[(业务 MySQL)] --> API[FastAPI 数据适配与指标服务]
    NWP[远程 NWP 文件源] --> JOB[定时同步与最近网格裁剪]
    JOB --> SNAP[(本地 NWP 快照)]
    SNAP --> API
    KB[(私有知识库)] --> AGENT[多智能体 / RAG]
    API --> AGENT
    API --> WEB[Vue 3 + ECharts]
    AGENT --> WEB
```

生产环境需要在平台之外补充反向代理、TLS、正式身份认证、权限控制、集中日志和密钥管理。智能体输出仅用于辅助决策，不能绕过调度授权与安全校核。
