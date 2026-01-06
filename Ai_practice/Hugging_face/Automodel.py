from transformers import AutoModel
from huggingface_hub import notebook_login

model = AutoModel.from_pretrained("bert-base-cased") 
model.save_pretrained("D:\\Learning\\ai\\Ai-learning\\Ai_practice\\Hugging_face\\bert-base-cased")
print("Model saved successfully.")

notebook_login() # 登录Hugging Face账号
model.push_to_hub("my-awesome-model")

