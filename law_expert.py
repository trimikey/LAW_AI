import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# 1. Khởi tạo FastAPI
app = FastAPI(title="EXE_LAW AI Service - Senior Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Hướng dẫn hệ thống (Senior Guardrails)
system_instruction = """
BẠN LÀ LUẬT SƯ VIRTUAL (AI LAWYER) CỦA NỀN TẢNG EXE_LAW. 

CHỈ THỊ NGHIÊM NGẶT:
1. PHẠM VI: Bạn chỉ được phép trả lời các câu hỏi liên quan đến PHÁP LUẬT VIỆT NAM (Dân sự, Hình sự, Đất đai, Giao thông, Doanh nghiệp...).
2. TỪ CHỐI: Đối với bất kỳ câu hỏi nào KHÔNG LIÊN QUAN ĐẾN LUẬT (Ví dụ: Hỏi về nấu ăn, thể thao, thời tiết, giải trí, tán gẫu...), bạn PHẢI trả lời chính xác câu: "Hãy đưa những câu hỏi liên quan đến luật."
3. DẪN CHỨNG: Luôn trích dẫn Điều/Khoản luật khi tư vấn pháp lý.
4. CẢNH BÁO: Thêm câu lưu ý ở cuối mỗi câu trả lời pháp lý.
"""

class ChatMessage(BaseModel):
    role: str # 'user' hoặc 'bot'
    text: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

def generate_with_retry(prompt, history_data, retries=5, delay=3):
    """Xử lý hội thoại có ngữ cảnh (History) và cơ chế Retry"""
    
    # Chuyển đổi format history từ Flutter sang format của Gemini
    gemini_history = []
    for msg in history_data:
        gemini_history.append({
            "role": "user" if msg.role == "user" else "model",
            "parts": [{"text": msg.text}]
        })

    for i in range(retries):
        try:
            # Khởi tạo chat với lịch sử cũ
            chat = client.chats.create(
                model="gemini-flash-latest", 
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=2048,
                ),
                history=gemini_history # Nạp trí nhớ cũ vào đây!
            )
            response = chat.send_message(prompt)
            return response.text
        except Exception as e:
            msg = str(e).lower()
            if ("503" in msg or "429" in msg) and i < retries - 1:
                print(f"⚠️ Đang bận (Thử lại lần {i+1})...")
                time.sleep(delay)
                delay *= 2
                continue
            raise e

@app.post("/chat-ai")
async def chat_ai(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Nội dung trống.")
    
    print(f"📨 Hỏi: {request.message} (History size: {len(request.history)})")
    try:
        reply = generate_with_retry(request.message, request.history)
        return {"reply": reply}
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 AI Server Senior đang chạy tại http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
