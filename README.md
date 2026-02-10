# 零售贷款智能运营系统（基于 OpenCode）

基于 **opencode-sdk** 的零售贷款智能运营系统：支持数据加载与查看、贷款意愿模型与推荐模型训练、个性化推荐与意愿预测；用户可通过界面按钮完成全流程，也可在智能对话中用自然语言下达「加载数据」「训练模型」「给某客户推荐产品」等指令。**主 Agent 必须通过 3 个子 Agent（data_agent、training_agent、recommendation_agent）完成所有业务操作**，子 Agent 调用 MCP 工具并流式回复。

## 技术栈

- **opencode-sdk**：通过 SDK 调用 opencode 服务器
- **Gradio**：Web UI 界面
- **scikit-learn**：机器学习模型
- **recommenders**：推荐系统框架

## 快速开始

### 仅 Web UI（推荐先做）

若只需使用**数据管理、模型训练、模型演示**，不需要 OpenCode 服务器：

1. **环境要求**：Python 3.10+，建议使用虚拟环境
2. **安装依赖**：在项目根目录执行 `pip install -r requirements.txt`
3. **启动**：`python main.py`
4. **访问**：浏览器打开 http://localhost:7860

可依次使用：
- **Tab 1「数据管理」**：加载/刷新示例数据、查看数据概况
- **Tab 2「模型训练」**：训练贷款意愿模型与推荐模型、查看训练细节
- **Tab 3「模型演示」**：选择客户、获取推荐、查看意愿评估

### 命令行 Agent（自然语言交互）

若要使用 `python -m scripts.run_agent` 做自然语言对话（如「加载数据，然后训练模型」），需要：

1. **安装并启动 OpenCode 服务器**（默认端口 4096）：
   ```bash
   # 安装 OpenCode（任选一种方式）
   npm i -g opencode-ai
   # 或
   brew install anomalyco/tap/opencode
   # 或参考：https://opencode.ai/download
   
   # 在项目根目录启动服务器
   opencode serve
   ```

2. **配置 OpenCode**：
   - 将 `opencode.json.example` 复制为项目根目录下的 `opencode.json`
   - 按需修改 `model`、`server.port`、`mcp` 等配置

3. **运行命令行 Agent**：
   ```bash
   python -m scripts.run_agent
   ```

4. **交互示例**：
   - 输入：`加载数据` → Agent 会委派 data_agent 调用 load_data
   - 输入：`训练模型` → Agent 会委派 training_agent 调用 train_models
   - 输入：`给客户ID为5的客户做贷款产品推荐` → Agent 会委派 recommendation_agent 调用 recommend
   - 输入 `exit` 或空回车结束对话

## 安装

在项目根目录执行：

```bash
pip install -r requirements.txt
```

**注意**：
- 若 `opencode-ai` 安装失败或版本不对，可暂时跳过，只装其余依赖（仅 Web UI 不需要 opencode-ai）
- 若仅用 Web UI，可执行：`pip install anyio gradio python-dotenv pandas scikit-learn numpy recommenders plotly`
- 若需命令行 Agent，可尝试：`pip install --pre opencode-ai`

## 配置

### 环境变量（可选）

在项目根目录创建 `.env` 文件，可覆盖默认值：

```env
OPENCODE_BASE_URL=http://localhost:4096
OPENCODE_MODEL_PROVIDER=deepseek
OPENCODE_MODEL_ID=deepseek-chat
HOST=0.0.0.0
PORT=7860
```

### OpenCode 配置

使用命令行 Agent 时，需要在 opencode 服务器中配置 MCP 服务器和子 Agent：

1. **MCP 服务器配置**：在 `opencode.json` 的 `mcp` 字段中注册 `loan_agent` MCP 服务器
   ```json
   "mcp": {
     "loan_agent": {
       "type": "local",
       "command": ["python", "-m", "tools.loan_agent_mcp_server"],
       "enabled": true
     }
   }
   ```

2. **主 Agent 工具限制**：在 `opencode.json` 的 `tools` 字段中禁用所有 MCP 工具（`mcp__loan_agent__*: false`），确保主 Agent 只能通过 Task 委派子 Agent

3. **子 Agent 配置**：在 `opencode.json` 的 `agent` 字段中注册三个子 Agent：
   - `data_agent`：负责数据加载与概况查询，可调用 `load_data`、`get_data_summary`、`list_customers`、`list_products`
   - `training_agent`：负责模型训练，可调用 `train_models`
   - `recommendation_agent`：负责推荐与意愿预测，可调用 `recommend`、`predict_propensity`、`similar_items`

参考 `opencode.json.example` 文件获取完整配置示例。工具定义见 `tools/loan_agent_tools.py` 中的 `get_loan_agent_tools()` 函数，子 Agent 定义见 `agents/definitions.py` 中的 `build_sub_agent_definitions()` 函数。

**工作流程**：主 Agent 接收用户请求 → 通过 Task 委派给相应的子 Agent → 子 Agent 调用 MCP 工具 → 返回结果给主 Agent → 主 Agent 汇总并回复用户。

## 架构验证

按下面顺序验证，全部通过即可认为架构正常：

### 验证 1：Web UI 全流程（不依赖 OpenCode）

**目的**：确认数据、训练、推荐链路在本项目内是通的。

1. 在项目根目录启动：`python main.py`
2. 浏览器打开：http://localhost:7860
3. 依次操作：
   - **Tab 1「数据管理」**：点击「加载/刷新示例数据」→ 等待完成 → 查看「数据概况」
   - **Tab 2「模型训练」**：点击「开始训练」→ 等待完成 → 查看训练指标
   - **Tab 3「模型演示」**：选择一名客户 → 点击「获取推荐」→ 查看推荐列表

**通过标准**：三步都能完成且无报错。

### 验证 2：MCP 服务器能独立启动

**目的**：确认 `tools.loan_agent_mcp_server` 和 8 个工具能被正确加载。

1. 在项目根目录执行：`python -m tools.loan_agent_mcp_server`
2. 观察 stderr 应出现：项目根目录、已加载 loan_agent_tools 共 8 个工具、正在创建 MCP 服务器、已注册 8 个工具等日志
3. 进程会挂起等待 stdin，这是预期行为。用 Ctrl+C 结束即可。

**通过标准**：有上述日志且无 Python 报错。

### 验证 3：OpenCode 服务 + opencode.json

**目的**：确认 OpenCode 能按项目配置启动并加载 MCP。

1. 确保项目根目录存在 `opencode.json`（可参考 `opencode.json.example`）
2. 在项目根目录启动 OpenCode 服务器：`opencode serve`
3. 确认无报错、服务监听在配置的端口（默认 4096）

**通过标准**：`opencode serve` 正常启动且不报配置错误。

### 验证 4：命令行 Agent 能对话并调用工具

**目的**：证明 opencode-sdk → OpenCode → 主 Agent → Task 委派 → 子 Agent → MCP 工具 整条链路通。

1. 保持 `opencode serve` 在运行
2. 在项目根目录执行：`python -m scripts.run_agent`
3. 出现 `你:` 后，依次输入：
   - `加载数据` → 应看到委派 data_agent，调用 load_data
   - `查看一下数据概况` → 应看到调用 get_data_summary
   - `训练模型` → 应看到委派 training_agent，调用 train_models
   - `给客户ID为5的客户做贷款产品推荐` → 应看到委派 recommendation_agent，调用 recommend
4. 输入 `exit` 或空回车结束

**通过标准**：至少能完成「加载数据」和「训练模型」两类指令并得到相关回复。

## 功能

1. **数据管理**：加载和查看示例数据（MovieLens 100k 转贷款场景）
2. **模型训练**：训练贷款意愿模型和推荐模型
3. **智能推荐**：为指定客户推荐贷款产品
4. **意愿预测**：预测客户对特定产品的申请意愿
5. **相似产品**：查找与某产品相似的其他产品
6. **客户/产品列表**：列出可用的客户ID和产品ID

## 常见问题

### 1. 导入错误：`No module named 'app'` 或 `No module named 'tools'`

**原因**：未在项目根目录运行。  
**解决**：所有命令都在项目根目录（即包含 `main.py`、`app/`、`tools/` 的目录）下执行。

### 2. 安装 `recommenders` 或 `opencode-ai` 失败

**recommenders**：需要较新版本的 pip，可先升级：`pip install -U pip`，再安装依赖。  
**opencode-ai**：若仅用 Web UI，可暂时不安装；若需命令行 Agent，可尝试 `pip install --pre opencode-ai` 或查阅 [OpenCode 文档](https://opencode.ai/docs)。

### 3. 运行 `scripts.run_agent` 报错连接被拒绝

**原因**：OpenCode 服务器未启动或地址/端口不对。  
**解决**：先在同一机器上执行 `opencode serve`，并确认 `OPENCODE_BASE_URL` 与 `opencode.json` 中的 `server.port` 一致。

### 4. 数据加载或训练很慢

首次加载会从 recommenders 拉取 MovieLens 100k 数据并做预处理，属正常现象；后续会使用缓存。

## 与 customer-behavior-ml-2 的区别

1. **Agent 执行方式**：
   - ml-2: 使用 `claude-agent-sdk` 直接调用 Claude Code CLI，Agent 在本地进程执行
   - ml-3: 使用 `opencode-sdk` 通过 REST API 调用 OpenCode 服务器，Agent 在服务器端执行

2. **子 Agent 配置**：
   - ml-2: 使用 `claude-agent-sdk` 的 `AgentDefinition` 在代码中定义子 Agent
   - ml-3: 子 Agent 定义在 `agents/definitions.py` 中提供，但需要在 `opencode.json` 配置文件中注册才能使用

3. **配置方式**：
   - ml-2: 通过 `core/config.py` 的 `LLM_BACKEND` 配置模型后端
   - ml-3: 通过 `opencode.json` 配置文件管理模型、MCP 服务器和子 Agent

4. **工具注册**：
   - ml-2: 使用 `claude-agent-sdk` 的 `create_sdk_mcp_server` 在代码中创建 MCP 服务器
   - ml-3: 需要在 `opencode.json` 中配置 MCP 服务器，工具定义通过 `get_loan_agent_tools()` 提供

5. **MCP 工具数量**：
   - ml-2: 6 个工具（load_data、get_data_summary、train_models、recommend、predict_propensity、similar_items）
   - ml-3: 8 个工具（新增 list_customers、list_products）

6. **工作流程设计**：
   - ml-2: 主 Agent 可以选择直接调用 MCP 工具或通过 Task 委派子 Agent（两种方式都支持）
   - ml-3: **主 Agent 必须通过 Task 委派子 Agent 完成所有业务操作**，不能直接调用 MCP 工具（强制通过子 Agent 的工作流程）

## 更多文档

- **[ARCHITECTURE.md](ARCHITECTURE.md)**：详细的架构设计图和说明