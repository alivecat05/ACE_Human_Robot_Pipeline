import os
import socket
import requests
import time

FEISHU_URL = os.environ.get("FEISHU_URL", "")
    

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 这里不需要真的连通，只是为了诱导系统选出出口 IP
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def main():
    if not FEISHU_URL:
        raise SystemExit("请先设置 FEISHU_URL 环境变量")
    print("正在监控网络状态...")
    while True:
        ip = get_local_ip()
        
        # 只要 IP 不是本地回环地址，说明网卡已就绪
        if ip != "127.0.0.1":
            try:
                msg = {
                    "msg_type": "text",
                    "content": {"text": f"IP 已上线: {ip}"}
                }
                res = requests.post(FEISHU_URL, json=msg, timeout=5)
                if res.status_code == 200:
                    print("推送成功，程序退出")
                    break
            except Exception as e:
                print(f"网络已分配 IP，但还无法连接飞书服务器: {e}")
        
        time.sleep(5) # 每 5 秒重试一次

if __name__ == "__main__":
    main()