from crewai.tools import BaseTool
from datetime import datetime

class CurrentTimeTool(BaseTool):
    name: str = "current_time"
    description: str = "获取当前日期和时间，返回格式如 '2026-04-20 15:30:45'。不需要任何输入参数。"

    def _run(self, query: str = "") -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")