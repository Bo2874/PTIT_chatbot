from vector_search import VectorSearchAgent, get_texts_by_ids
from indexer import create_vectorstore
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import OPENAI_API_KEY
from rewrite_query import rewrite_query_for_vectorsearch

async def streaming_chatbot(query: str, history: list[dict]):
    URI = "http://localhost:19530"
    collection_name = "ptit_vectors"

    # Tạo vector store và truy xuất thông tin
    vector_store = create_vectorstore(URI, collection_name)
    retriever = VectorSearchAgent(vector_store)
    print(rewrite_query_for_vectorsearch(query, history))
    ids = retriever.retrieve(rewrite_query_for_vectorsearch(query, history), 4)
    infors = get_texts_by_ids(ids)

    print(infors)

    # Tạo các message khởi tạo
    system_msg = SystemMessage(
        content="Bạn là 1 người tư vấn về các thông tin của Học viện Công nghệ Bưu chính Viễn Thông PTIT."
    )

    # Chuyển history dạng dict -> message
    history_messages = []
    for msg in history[-4:]:  # chỉ lấy 4 tin nhắn gần nhất
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    # Tạo truy vấn người dùng
    human_msg = HumanMessage(
        content=(
            f"""**Đây là truy vấn của người dùng:** {query}\n\n
                **Đây là thông tin đã truy xuất được từ cơ sở dữ liệu:**\n{infors}\n\n
                Dựa vào thông tin trên, bạn hãy trả lời câu hỏi từ người dùng một cách đầy đủ, rõ ràng và thân thiện, gợi ý người dùng truy cập vào link để xem chi tiết, khi gợi ý bằng link bắt buộc phải xuống dòng.
                Nếu thấy truy vấn của người dùng là chào hỏi hoặc nói chuyện phiếm thì không cần dùng đến thông tin và link đã truy suất, tự đưa ra phản hồi theo truy vấn người dùng.
                Chỉ chào người dùng nếu người dùng chào, nếu người dùng hỏi thì chỉ trả lời không cần thêm 'Chào bạn!' vào đầu câu.
                """
        )
    )

    # Tạo prompt
    messages = [system_msg] + history_messages + [human_msg]

    # Khởi tạo model có stream
    llm = ChatOpenAI(
        model="gpt-5-nano-2025-08-07",  #gpt-5-nano-2025-08-07
        temperature=1,
        openai_api_key=OPENAI_API_KEY,
        streaming = True
    )

    # Stream token
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content     #Khi dùng return hàm sẽ kết thúc ngay lập tức, trả về 1 giá trị duy nhất
                                    # yeild hàm tạm dừng, trả về 1 giá trị, và có thể tiếp tục từ chỗ đã dừng vào lần gọi tiếp theo
