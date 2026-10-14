# 发布检查

- 默认全部问题：start-index=0、max-items=0。
- 默认五种方法：disco_full、without_rst、without_graph、without_planning、standard_rag。
- 五组同一 top-k 和最终生成预算，不提供人为降配演示入口。
- 仅包含源码、说明和依赖；不复制数据、权重、索引、结果、日志、虚拟环境和旧Git历史。
- `.env.example` 仅占位值；真实 API key 不得提交。
- 用 `python -m pytest tests -q` 检查测试。
- 实际全量推理需要用户提供数据、模型及API服务；发布整理不会自动启动该实验。
- 上游许可证待核实：目前不能声称已具备公开再分发授权。
- 未向GitHub推送，未创建远端仓库。
