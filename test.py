import requests
import time
import json

# 提交分析任务
resp = requests.post("http://localhost:8000/analyze", json={"product_name": "小米手环9"})
if resp.status_code != 200:
    print(f"提交失败: {resp.status_code} - {resp.text}")
    exit(1)

data = resp.json()
task_id = data["task_id"]
print(f"任务已提交，task_id: {task_id}")

# 轮询结果
while True:
    resp = requests.get(f"http://localhost:8000/result/{task_id}")
    data = resp.json()
    if data["status"] == "completed":
        print("\n最终报告：")
        print(json.dumps(data["result"], indent=2, ensure_ascii=False))
        break
    else:
        print("处理中...")
        time.sleep(2)