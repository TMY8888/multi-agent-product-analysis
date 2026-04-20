import os
import uuid
import threading
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, TypedDict, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from loguru import logger
from langgraph.graph import StateGraph, END
from crewai import Agent, Task, Crew, Process, LLM
import redis
from tools import CurrentTimeTool
load_dotenv()

# ---------- 配置 ----------
class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-plus")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

settings = Settings()

logger.add("logs/app.log", rotation="10 MB", level="INFO",
           format="{time} | {level} | {extra[correlation_id]} | {message}")
logger.configure(extra={"correlation_id": "N/A"})

# ---------- Redis 连接（可选）---------
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connected")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
    redis_client = None

# ---------- 结构化输出模型 ----------
class ReportOutput(BaseModel):
    product_name: str = Field(description="产品名称")
    features: List[str] = Field(description="核心功能列表")
    price: str = Field(description="价格信息")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")
    competitors: List[str] = Field(description="主要竞品列表")
    summary: str = Field(description="购买建议总结")

# ---------- 动态 LLM（支持多模型切换，无tools参数）---------
def get_llm():
    return LLM(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=0.7
    )

# ---------- CrewAI Agent（不使用任何自定义工具，依赖模型内置搜索）---------
def create_researcher():
    time_tool = CurrentTimeTool()
    return Agent(
        role="市场研究员",
        goal="搜索产品信息（功能、价格、评价、竞品）",
        backstory="""你擅长使用搜索引擎获取准确信息。在开始搜索前，请先调用 current_time 工具获取当前时间，记录在报告中。""",
        tools=[time_tool],
        llm=get_llm(),
        verbose=True
    )

def create_analyst():
    return Agent(
        role="产品分析师",
        goal="根据信息生成JSON格式的分析报告",
        backstory="你必须严格按照JSON格式输出，包含产品名称、功能列表、价格、优缺点、竞品、总结。",
        llm=get_llm(),
        verbose=True
    )

def create_research_task(agent, product_name: str) -> Task:
    return Task(
        description=f"""
        步骤1：调用 current_time 工具获取当前时间。
        步骤2：搜索产品「{product_name}」的核心功能、价格、用户评价和主要竞品。
        """,
        expected_output="一个Markdown列表，包含当前时间、功能、价格、评价、竞品。",
        agent=agent
    )

def create_analysis_task(agent, product_name: str, context: str) -> Task:
    return Task(
        description=f"""
        基于以下搜索结果，为「{product_name}」生成一份分析报告。
        搜索结果：{context}
        报告中必须包含以下字段：
        - product_name: 产品名称
        - features: 核心功能列表（字符串数组）
        - price: 价格信息
        - pros: 优点列表（字符串数组）
        - cons: 缺点列表（字符串数组）
        - competitors: 主要竞品列表（字符串数组）
        - summary: 购买建议总结
        """,
        expected_output="一个合法的JSON对象，包含上述所有字段。",
        agent=agent,
        output_pydantic=ReportOutput   # 关键：强制输出为 Pydantic 模型
    )

# ---------- LangGraph 状态定义 ----------
class AnalysisState(TypedDict):
    product_name: str
    search_result: Optional[str]
    report_json: Optional[str]
    report_obj: Optional[ReportOutput]
    retry_count: int
    correlation_id: str

# ---------- LangGraph 节点函数 ----------
def search_node(state: AnalysisState) -> AnalysisState:
    cid = state["correlation_id"]
    product = state["product_name"]
    logger.bind(correlation_id=cid).info(f"[LangGraph] 搜索节点: {product}")
    researcher = create_researcher()
    research_task = create_research_task(researcher, product)
    crew = Crew(agents=[researcher], tasks=[research_task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    state["search_result"] = str(result)
    return state


def analyze_node(state: AnalysisState) -> AnalysisState:
    cid = state["correlation_id"]
    product = state["product_name"]
    context = state.get("search_result", "")
    logger.bind(correlation_id=cid).info("[LangGraph] 分析节点")
    analyst = create_analyst()
    analysis_task = create_analysis_task(analyst, product, context)
    crew = Crew(agents=[analyst], tasks=[analysis_task], process=Process.sequential, verbose=False)
    result = crew.kickoff()

    # 优先从 result.pydantic 获取结构化输出
    if hasattr(result, 'pydantic') and result.pydantic:
        report_obj = result.pydantic
        state["report_obj"] = report_obj
        state["report_json"] = report_obj.json()
        logger.bind(correlation_id=cid).info("成功使用 output_pydantic 获取结构化报告")
    else:
        # 兼容旧逻辑：尝试从文本解析 JSON
        output_str = str(result)
        if "```json" in output_str:
            output_str = output_str.split("```json")[1].split("```")[0].strip()
        elif "```" in output_str:
            output_str = output_str.split("```")[1].split("```")[0].strip()
        state["report_json"] = output_str
        try:
            report_obj = ReportOutput.parse_raw(output_str)
            state["report_obj"] = report_obj
        except Exception as e:
            logger.bind(correlation_id=cid).error(f"JSON解析失败: {e}, 原始: {output_str}")
            state["report_obj"] = ReportOutput(
                product_name=product,
                features=["解析失败"],
                price="暂无",
                pros=["暂无"],
                cons=["暂无"],
                competitors=["暂无"],
                summary="报告生成失败，请稍后重试。"
            )
    return state

def should_retry(state: AnalysisState) -> bool:
    retry = (state["retry_count"] < 2 and
             (len(state["report_obj"].features) == 0 or
              "解析失败" in state["report_obj"].features[0]))
    if retry:
        logger.bind(correlation_id=state["correlation_id"]).warning(f"质量不达标，重试 ({state['retry_count']+1}/2)")
    return retry

def retry_node(state: AnalysisState) -> AnalysisState:
    state["retry_count"] += 1
    return state

def finalize_node(state: AnalysisState) -> AnalysisState:
    logger.bind(correlation_id=state["correlation_id"]).info("最终节点，报告生成完成")
    return state

def build_graph():
    workflow = StateGraph(AnalysisState)
    workflow.add_node("search", search_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("search")
    workflow.add_edge("search", "analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_retry,
        {
            True: "retry",
            False: "finalize"
        }
    )
    workflow.add_edge("retry", "analyze")
    workflow.add_edge("finalize", END)
    return workflow.compile()

# ---------- 任务存储（Redis + 内存回退）---------
def save_task_result(task_id: str, report: ReportOutput):
    data = json.dumps(report.dict(), ensure_ascii=False)
    if redis_client:
        redis_client.setex(f"task:{task_id}", 3600, data)
    else:
        _memory_tasks[task_id] = data

def get_task_result(task_id: str) -> Optional[ReportOutput]:
    if redis_client:
        data = redis_client.get(f"task:{task_id}")
    else:
        data = _memory_tasks.get(task_id)
    if data:
        return ReportOutput(**json.loads(data))
    return None

_memory_tasks = {}

# ---------- WebSocket 管理 ----------
active_websockets: Dict[str, WebSocket] = {}

async def notify_websocket(task_id: str, message: dict):
    ws = active_websockets.get(task_id)
    if ws:
        try:
            await ws.send_json(message)
        except:
            pass

# 全局事件循环引用
main_event_loop = None

def run_analysis_sync(product_name: str, correlation_id: str, task_id: str):
    global main_event_loop
    logger.bind(correlation_id=correlation_id).info(f"开始分析: {product_name}")
    if main_event_loop:
        asyncio.run_coroutine_threadsafe(
            notify_websocket(task_id, {"type": "status", "status": "searching", "message": "正在搜索产品信息..."}),
            main_event_loop
        )
    graph = build_graph()
    initial_state = {
        "product_name": product_name,
        "search_result": None,
        "report_json": None,
        "report_obj": None,
        "retry_count": 0,
        "correlation_id": correlation_id
    }
    final_state = graph.invoke(initial_state)
    report = final_state["report_obj"]
    save_task_result(task_id, report)
    if main_event_loop:
        asyncio.run_coroutine_threadsafe(
            notify_websocket(task_id, {"type": "completed", "result": report.dict()}),
            main_event_loop
        )
    logger.bind(correlation_id=correlation_id).info("分析完成")

# ---------- FastAPI 应用 ----------
app = FastAPI(title="企业级多智能体分析系统", version="4.0.0")

@app.on_event("startup")
async def startup_event():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    logger.info("Main event loop captured for WebSocket notifications")

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    with logger.contextualize(correlation_id=correlation_id):
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

class AnalyzeRequest(BaseModel):
    product_name: str

class AnalyzeResponse(BaseModel):
    task_id: str
    status: str
    message: str

class ResultResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[ReportOutput] = None

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks, req: Request):
    task_id = str(uuid.uuid4())
    correlation_id = req.headers.get("X-Correlation-ID", task_id)
    background_tasks.add_task(run_analysis_sync, request.product_name, correlation_id, task_id)
    return AnalyzeResponse(
        task_id=task_id,
        status="processing",
        message="任务已提交，请使用 /result/{task_id} 查询结果，或通过 WebSocket 接收实时进度"
    )

@app.get("/result/{task_id}", response_model=ResultResponse)
async def get_result(task_id: str):
    report = get_task_result(task_id)
    if report:
        return ResultResponse(task_id=task_id, status="completed", result=report)
    else:
        return ResultResponse(task_id=task_id, status="pending", result=None)

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    active_websockets[task_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        active_websockets.pop(task_id, None)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/models")
async def list_models():
    return {
        "available_models": ["glm-4-plus", "glm-4-flash", "qwen-plus", "qwen-turbo"],
        "current_model": settings.LLM_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)