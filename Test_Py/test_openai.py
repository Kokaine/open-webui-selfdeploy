from openai import OpenAI 

client = OpenAI(
    api_key="c20ad0564bf04802a49cb1dee09270a5.Iu9V2RHaIa2R2n79",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
) 

response = client.images.generate(
    model="cogView-4-250304", #填写需要调用的模型编码
    prompt="在干燥的沙漠环境中，一棵孤独的仙人掌在夕阳的余晖中显得格外醒目。这幅油画捕捉了仙人掌坚韧的生命力和沙漠中的壮丽景色，色彩饱满且表现力强烈。",
    size="1440x720"
)

print(response.data[0].url)