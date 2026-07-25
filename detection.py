import base64
import os
import json
from openai import OpenAI

def image_to_base64(image_path):
    """将本地图片转换为Base64编码"""
    with open(image_path, "rb") as image_file:
        # 读取图片内容并转换为Base64编码
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")
        # 根据图片格式返回对应的data URL格式
        if image_path.lower().endswith(('.png', '.PNG')):
            return f"data:image/png;base64,{base64_str}"
        else:  # 默认jpeg格式
            return f"data:image/jpeg;base64,{base64_str}"

def process_images_in_folder(folder_path):
    """处理文件夹中的所有图片并保存结果"""
    # 初始化客户端
    client = OpenAI( 
        base_url='https://api-inference.modelscope.cn/v1',
        api_key='',  # 替换为你的ModelScope Token
    )
    
    # 获取文件夹中所有图片文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    for filename in os.listdir(folder_path):
        # 检查是否为图片文件
        if filename.endswith(image_extensions):
            # 提取编号（假设文件名格式为 crsim_数字_processed.扩展名）
            if filename.startswith('crsim_') and '_processed' in filename:
                try:
                    # 解析编号（例如从crsim_0_processed.png中提取0）
                    prefix = 'crsim_'
                    suffix = '_processed'
                    num_str = filename[len(prefix):filename.index(suffix)]
                    image_num = int(num_str)
                except (ValueError, IndexError):
                    print(f"文件名格式不符合要求，跳过: {filename}")
                    continue
                
                # 构建完整图片路径
                image_path = os.path.join(folder_path, filename)
                print(f"正在处理图片: {image_path}")
                
                # 转换图片为Base64格式的URL
                image_url = image_to_base64(image_path)
                
                # 调用模型
                response = client.chat.completions.create(
                    model='stepfun-ai/step3',  # ModelScope模型ID
                    messages=[{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': '请你给出图中的工业用地的bbox坐标，坐标值范围为0-512，并输出为json格式。'},
                            {'type': 'image_url', 'image_url': {'url': image_url}}
                        ],
                    }],
                    stream=True
                )
                
                # 收集模型输出结果
                full_response = []
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response.append(content)
                        print(content, end='', flush=True)  # 控制台输出
                
                # 组合完整响应并尝试解析为JSON
                try:
                    result_content = ''.join(full_response)
                    # 尝试将结果解析为JSON（确保格式正确）
                    json_data = json.loads(result_content)
                except json.JSONDecodeError:
                    print(f"\n警告：{filename} 的模型输出不是有效的JSON格式，将直接保存原始内容")
                    json_data = {"raw_response": result_content}
                
                # 构建输出JSON文件路径（与图片同目录）
                output_json = os.path.join(folder_path, f"predicted_{image_num}.json")
                
                # 保存结果到JSON文件
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)
                
                print(f"\n结果已保存到: {output_json}\n")

# 图片所在文件夹路径（请根据实际情况修改）
image_folder = ""

# 处理文件夹中的所有图片
process_images_in_folder(image_folder)
