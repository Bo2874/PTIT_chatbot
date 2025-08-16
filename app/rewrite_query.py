from config import OPENAI_API_KEY
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


def rewrite_query_for_vectorsearch(query: str, history: list[dict]):
    # Chuyển history dạng dict -> message
    history_messages = []
    for msg in history[-4:]:  # chỉ lấy 4 tin nhắn gần nhất
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))


    system_msg = SystemMessage(
        content = 'Bạn là 1 chuyên gia viết lại truy vấn người dùng dựa vào lịch sử chat. Truy vấn này dùng để tìm kiếm các tài liệu liên quan đến PTIT.'
    )
    human_msg = HumanMessage(
        content = f"""
        **Đây là truy vấn gần nhất từ người dùng:**{query}.
        Dựa vào lịch sử chat hãy bổ viết lại truy vấn người dùng cho đủ ý để thực hiện vector search (Lưu ý: Không được bịa thêm).
        Với những câu truy vấn tương tự như sau:
        - Xin chào
        - Chào bạn
        - Bạn là ai
        Thì giữ nguyên câu truy vấn của người dùng và không cần tóm tắt hay viết lại câu truy vấn đó.
        Chỉ trả về 1 câu truy vấn duy nhất, không giải thích gì thêm.
        """
    )

    # Tạo prompt
    messages = [system_msg] + history_messages + [human_msg]

    # Khởi tạo model có stream
    llm = ChatOpenAI(
        model="gpt-5-nano-2025-08-07",  #gpt-5-nano-2025-08-07
        temperature=1,
        openai_api_key=OPENAI_API_KEY
    )

    res = llm.invoke(messages)
    return res.content