import os
from openai import OpenAI

# 构造 client
client = OpenAI(
    api_key=os.environ.get("HUNYUAN_API_KEY"), # 混元 APIKey
    base_url="https://api.hunyuan.cloud.tencent.com/v1", # 混元 endpoint
)


# 自定义参数传参示例
completion = client.chat.completions.create(
    model="hunyuan-vision",
    messages=[
        {
            "role": "user",
            "contents": [
                {
                    "type": "text",
                    "text": "这是什么图片"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qcloudimg.tencent-cloud.cn/raw/42c198dbc0b57ae490e57f89aa01ec23.png"
                    }
                }
            ]
        },
    ],
)