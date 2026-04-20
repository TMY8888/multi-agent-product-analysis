import streamlit as st
import requests
import time
import json

st.set_page_config(page_title="产品分析系统", page_icon="🤖", layout="centered")
st.title("🤖 多智能体产品分析系统")
st.markdown("输入产品名称，系统将自动搜索并生成结构化分析报告")

# 后端 API 地址（部署后修改为实际地址）
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://47.106.110.30:8000")

product_name = st.text_input("产品名称", placeholder="例如：小米手环9、华为Mate60")

if st.button("开始分析", type="primary"):
    if not product_name.strip():
        st.error("请输入产品名称")
    else:
        # 提交任务
        with st.spinner("正在提交分析任务..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/analyze",
                    json={"product_name": product_name},
                    timeout=10
                )
                resp.raise_for_status()
                task_id = resp.json()["task_id"]
                st.success(f"任务已提交，ID: {task_id}")
            except Exception as e:
                st.error(f"提交失败: {e}")
                st.stop()

        # 轮询结果
        progress_bar = st.progress(0, text="分析中...")
        status_placeholder = st.empty()
        result_placeholder = st.empty()
        max_wait = 60
        start = time.time()

        while time.time() - start < max_wait:
            try:
                resp = requests.get(f"{API_BASE_URL}/result/{task_id}", timeout=5)
                resp.raise_for_status()
                data = resp.json()
                if data["status"] == "completed":
                    progress_bar.progress(100)
                    status_placeholder.success("✅ 分析完成！")
                    report = data["result"]
                    with result_placeholder.container():
                        st.markdown("## 📊 分析报告")
                        st.markdown(f"### {report['product_name']}")
                        st.markdown("**核心功能**")
                        for f in report['features']:
                            st.markdown(f"- {f}")
                        st.markdown(f"**价格**：{report['price']}")
                        st.markdown("**优点**")
                        for p in report['pros']:
                            st.markdown(f"- {p}")
                        st.markdown("**缺点**")
                        for c in report['cons']:
                            st.markdown(f"- {c}")
                        st.markdown("**主要竞品**")
                        for comp in report['competitors']:
                            st.markdown(f"- {comp}")
                        st.markdown("**购买建议**")
                        st.info(report['summary'])
                    break
                else:
                    elapsed = int(time.time() - start)
                    progress_bar.progress(min(elapsed / max_wait, 0.95))
                    status_placeholder.info(f"分析中... ({elapsed}秒)")
                    time.sleep(2)
            except Exception as e:
                status_placeholder.error(f"查询失败: {e}")
                break
        else:
            status_placeholder.error("分析超时，请稍后重试。")