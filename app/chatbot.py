import traceback
from vector_search import VectorSearchAgent, get_texts_by_ids
from rerank import Reranker
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from config import OPENAI_API_KEY
from rewrite_query import rewrite_query_for_vectorsearch
from milvus_connection import vector_store
from logging_config import setup_logging
import time

logger = setup_logging()

# Khởi tạo model có stream
llm = ChatOpenAI(
    model="gpt-4.1-mini-2025-04-14",
    temperature=1,
    openai_api_key=OPENAI_API_KEY,
    streaming=True
)

async def streaming_chatbot(query: str, history: list[dict]):
    # ---- BỌC TOÀN BỘ HÀM TRONG TRY-EXCEPT TỔNG QUÁT ----
    # Để đảm bảo không có lỗi nào không bị bắt
    try:
        # === Giai đoạn 1: Viết lại câu truy vấn ===
        try:
            logger.info(f"Original query: {query}")
            start_time = time.time()
            new_query = rewrite_query_for_vectorsearch(query, history)
            logger.info(f"Rewritten query: {new_query}")
            end_time = time.time()
            logger.info(f"Thời gian rewrite: {(end_time - start_time):.2f} giây")
        except Exception as e:
            logger.error(f"LỖI trong quá trình rewrite query: {e}")
            logger.error(traceback.format_exc()) # In ra chi tiết lỗi
            yield "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý yêu cầu của bạn. Vui lòng thử lại sau."
            return # Dừng hàm nếu có lỗi ở bước này

        # === Giai đoạn 2: Truy xuất vector ===
        infors = [] # Khởi tạo giá trị mặc định
        try:
            start_time1 = time.time()
            retriever = VectorSearchAgent(vector_store)
            ids = retriever.retrieve(new_query, 8)
            infors = get_texts_by_ids(ids)
            end_time1 = time.time()
            logger.info(f"Thời gian truy xuất vector: {(end_time1 - start_time1):.2f} giây")
            
            # # Sử dụng reranker để sắp xếp lại kết quả
            # start_time_rerank = time.time()
            # reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")
            # reranked_docs = reranker.rerank(new_query, [Document(page_content=info['text'], metadata=info) for info in infors], top_k=5)
            # end_time_rerank = time.time()
            # logger.info(f"Thời gian rerank: {(end_time_rerank - start_time_rerank):.2f} giây")

            formatted_docs = [] # Formatted documents for prompt
            infors_text = "" # Default text if no info
            for doc in reranked_docs:
                url = doc.metadata.get("url", "")
                snippet = doc.page_content.strip()
                if url:
                    formatted_docs.append(f"- {snippet} (Nguồn: [{url}]({url}))")
                else:
                    formatted_docs.append(f"- {snippet}")

            infors_text = "\n".join(formatted_docs)
            print(infors_text)

        except Exception as e:
            logger.error(f"LỖI trong quá trình truy xuất vector: {e}")
            logger.error(traceback.format_exc())
            # Vẫn có thể tiếp tục mà không có thông tin bổ sung
            infors = "Không tìm thấy thông tin." # Hoặc để trống

        # === Giai đoạn 3: Tạo prompt và gọi LLM Streaming ===
        system_msg = SystemMessage(
            content="Bạn là 1 người tư vấn về các thông tin của Học viện Công nghệ Bưu chính Viễn Thông PTIT."
        )

        history_messages = []
        for msg in history[-4:]:
            role = msg.get("role") # Dùng .get() để tránh lỗi nếu key không tồn tại
            content = msg.get("content")
            if role == "user":
                history_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                history_messages.append(AIMessage(content=content))

        human_msg = HumanMessage(
            content=(
                f"""**Đây là truy vấn của người dùng:** {new_query}\n\n
                **Đây là thông tin đã truy xuất được từ cơ sở dữ liệu:**\n{infors}\n\n
                Nếu thông tin có liên quan đến truy vấn thì dựa vào thông tin trên, bạn hãy trả lời câu hỏi từ người dùng một cách đầy đủ, rõ ràng và thân thiện. Không cần thêm 'Chào bạn!' vào đầu câu trả lời.
                Nếu không có kết quả, trả về: "Hiện tại chúng tôi chưa cập nhật dữ liệu mà bạn đang hỏi, bạn có thể hỏi câu hỏi khác để mình giúp đỡ".
                Nếu khách hàng có chào hỏi thì cần chào hỏi lại khách hàng với vai trò là 1 người tư vấn về các thông tin của Học viện Công nghệ Bưu chính Viễn Thông PTIT.
                
                **Yêu cầu định dạng**
                *Câu đầu tiên là câu dẫn.
                *Sử dụng in đậm (**) cho các từ khóa quan trọng.
                *Khi trả lời, nếu cần đưa liên kết, hãy hiển thị dưới dạng liên kết ẩn với cú pháp [Tên hiển thị].Ví dụ: https://ptit.edu.vn/gioi-thieu/co-cau-to-chuc/hoi-dong-giao-su-co-so → [Hội đồng giáo sư cơ sở]
                **Ở cuối phản hồi thêm đoạn sau bằng dạng chữ in đậm: Bạn có thể truy cập vào website của Học viện hoặc liên hệ trực tiếp với phòng giáo vụ của trường để nhận được những thông tin cập nhật mới nhất! 
                """
            )
        )
        
        messages = [system_msg] + history_messages + [human_msg]
        
        first_token_received = False
        start_time2 = time.time()
        
        try:
            async for chunk in llm.astream(messages):
                if chunk.content:
                    if not first_token_received:
                        end_time2 = time.time()
                        elapsed_time = end_time2 - start_time2
                        logger.info(f"Thời gian nhận token đầu tiên (TTFT): {elapsed_time:.2f} giây")
                        first_token_received = True
                    yield chunk.content           
        except Exception as e:
            logger.error(f"LỖI trong quá trình streaming từ LLM: {e}")
            logger.error(traceback.format_exc())
            yield "Xin lỗi, đã có lỗi xảy ra khi kết nối đến AI. Vui lòng thử lại."

    except Exception as e:
        logger.error(f"LỖI TỔNG QUÁT không xác định: {e}")
        logger.error(traceback.format_exc())
        yield "Đã có một lỗi không mong muốn xảy ra. Chúng tôi đang kiểm tra."

# async def main():
#     query = "Học viện có các ngành nào?" # Thử một câu hỏi cụ thể hơn
#     history = []
    
#     # Lặp qua generator để nhận từng phần của câu trả lời
#     print("AI trả lời:")
#     async for chunk in streaming_chatbot(query, history):
#         print(chunk, end="", flush=True)
#     print("\n")

# if __name__ == "__main__":
#     asyncio.run(main())