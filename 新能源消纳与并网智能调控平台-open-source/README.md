# 新能源消纳与并网智能调控平台

面向风电、光伏场站的新能源运行监测与辅助决策平台。项目打通实时功率、历史气象观测、NWP 预报、超短期功率预测、消纳分析与多智能体问答链路，以数字化大屏呈现新能源并网运行态势。

> 本仓库仅提供程序源码。业务数据库、场站数据、NWP 原始文件、知识库语料、模型权重及任何访问凭据均不随仓库发布。数据合作或样例数据申请请联系 [ts26170007a31@cumt.edu.cn](mailto:ts26170007a31@cumt.edu.cn)。

## 功能模块

- **功率监测大屏**：场站地图、实时功率、理论/可利用功率、限电状态、历史实测与未来 4 小时超短期预测。
- **气象观测大屏**：测风塔与辐照度实测、24 小时趋势、NWP 最新起报、UTC 到 UTC+8 对齐及场站最近网格裁剪。
- **消纳分析大屏**：实际/理论/可利用功率对比、场站消纳与弃电指标、MAE、RMSE、ACC 和误差曲线。
- **智能调控中心**：语义理解、实时数据读取、知识检索、策略评估和报告生成；建议保留人工确认，不直接替代调度指令。
- **后台管理**：场站启停、数据源状态、服务器连接元数据与审计记录。

## 技术架构

| 层级 | 技术 |
| --- | --- |
| 前端 | Vue 3、TypeScript、Vite、ECharts、Element Plus、Pinia |
| 后端 | FastAPI、Pydantic Settings、PyMySQL、Pandas |
| 数据 | MySQL 业务库、SSH/NWP 文件源、本地知识库（均由部署方配置） |
| 智能分析 | RAG、知识图谱、多智能体编排、可配置的 OpenAI 兼容模型接口 |

## 目录结构

```text
.
├─ backend/                 # FastAPI 服务、领域模型、数据适配器和测试
│  ├─ app/
│  ├─ rag_core/             # RAG 与评测核心代码（不含语料/权重）
│  ├─ data/powergridqa/     # 私有知识库放置目录
│  └─ tests/
├─ frontend/                # Vue 3 可视化大屏与管理端
├─ scripts/                 # 最新 NWP 选择、下载与场站裁剪脚本
├─ data/weather/            # NWP 运行数据目录（不纳入版本控制）
└─ docs/                    # 架构、数据接入与安全说明
```

## 快速开始

### 1. 环境要求

- Python 3.11+
- Node.js 20+
- MySQL 8.0+（使用真实业务数据时）
- OpenSSH Client（接入远程 NWP 时）
- ecCodes（解析 GRIB 文件时，安装方式取决于操作系统）

### 2. 启动后端

```powershell
cd backend
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,nwp]"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

然后访问 `http://127.0.0.1:8001/docs` 查看接口文档。请先在 `.env` 中填写自己的数据库、SSH 与模型服务配置；不要提交该文件。

### 3. 启动前端

```powershell
cd frontend
npm ci
npm run dev
```

开发服务器默认监听 `http://127.0.0.1:5173`，并将 `/api` 代理到后端 `8001` 端口。

### 4. 接入 NWP

在本机 `~/.ssh/config` 中配置仅属于部署环境的 SSH Host 别名和密钥路径，再把别名写入 `backend/.env` 的 `NWP_SSH_HOST`。仓库不保存主机 IP、账号、跳板机、私钥或远程真实目录。

后端可按 `NWP_REFRESH_HOURS` 在北京时间定时更新，也可手动运行：

```powershell
python scripts/extract_latest_nwp.py --remote nwp-host --remote-dir /path/to/nwp/48h
```

## 数据与知识库

代码中的适配器展示了预期字段和表结构，但本仓库不提供生产数据、数据集或专有知识库。详细接入原则见 [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md)。如需数据合作，请联系 [ts26170007a31@cumt.edu.cn](mailto:ts26170007a31@cumt.edu.cn)。

## 安全说明

- 所有密钥、密码、Token、数据库地址和 SSH 配置只允许保存在本地 `.env` 或密钥管理服务中。
- 管理端默认密码仅用于本地开发，部署前必须替换，并建议接入正式身份认证与权限系统。
- 智能调控输出属于辅助决策，生产环境应实施人工复核、权限校验、审计追踪和安全边界验证。
- 若发现安全问题，请按照 [SECURITY.md](SECURITY.md) 私下报告，不要公开披露凭据或生产数据。

## 测试与构建

```powershell
cd backend
python -m pytest tests -q

cd ..\frontend
npm run build
```

## 许可证

源代码采用 [MIT License](LICENSE)。第三方数据、模型、知识文档及外部服务分别受其自身许可与使用条款约束。
