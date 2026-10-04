import streamlit as st
import asyncio
import json
from pathlib import Path
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools

# ============================================================
# REMOTE MCP AGENT
# ============================================================

async def run_remote_agent(
    mcp_url,
    api_key,
    model_name,
    prompt
):
    mcp_ctx = None
    session_ctx = None

    try:
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key
        )

        mcp_ctx = streamablehttp_client(
            mcp_url,
            timeout=60,
            sse_read_timeout=120
        )

        read_stream, write_stream, _ = await mcp_ctx.__aenter__()

        session_ctx = ClientSession(
            read_stream,
            write_stream
        )

        session = await session_ctx.__aenter__()
        await session.initialize()

        mcp_tools = await load_mcp_tools(session)

        agent = create_agent(
            model=model,
            tools=mcp_tools
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        messages = result.get("messages", [])

        for message in reversed(messages):
            if getattr(message, "type", None) == "ai":
                return message.content

        return "No AI response returned."

    finally:
        if session_ctx is not None:
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception:
                pass

        if mcp_ctx is not None:
            try:
                await mcp_ctx.__aexit__(None, None, None)
            except Exception:
                pass


# Helper to invoke MCP tool for document reading
async def call_mcp_rag(
    directory: str,
    question: str,
    api_key: str,
    embedding_model: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    temperature: float,
    n_results: int
):
    async with streamablehttp_client(
        "http://rag-server:8001/mcp"
    ) as (read_stream, write_stream, _):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                "run_rag",
                {
                    "directory": directory,
                    "question": question,
                    "api_key": api_key,
                    "embedding_model": embedding_model,
                    "model_name": model_name,
                    "chunk_size": int(chunk_size),
                    "chunk_overlap": int(chunk_overlap),
                    "temperature": float(temperature),
                    "n_results": int(n_results)
                }
            )

            if result.isError:
                raise RuntimeError(
                    f"MCP RAG tool failed: {result.content}"
                )

            # FastMCP may return the dictionary as structured content
            if getattr(result, "structuredContent", None):
                return result.structuredContent

            # Otherwise parse the text content
            if result.content:
                raw_text = result.content[0].text

                if raw_text:
                    return json.loads(raw_text)

            raise RuntimeError(
                f"MCP returned no usable result: {result}"
            )

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Intelligence")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Configuration")

mcp_mode = st.sidebar.radio(
    "MCP Mode",
    ["Local MCP", "Remote MCP"]
)

# ============================================================
# LOCAL MCP
# ============================================================

if mcp_mode == "Local MCP":

    st.header("🏠 Local MCP")
    st.subheader("🔍 Drive Search")

    st.markdown(
        """
        <div style="font-size:18px; line-height:1.6; margin-bottom:25px;">
        Searches recursively through a user-provided directory, checking both file names and contents.
        </div>
        """,
        unsafe_allow_html=True
    )

    directory = st.text_input("Directory", placeholder=r"D:\Bank_Statements")
    query = st.text_input("Keyword", placeholder="costco")
    max_results = st.number_input("Maximum Results", min_value=1, max_value=100, value=20)

    if st.button("🔍 Search", use_container_width=True):
        if not directory or not query:
            st.warning("Please enter both a directory and keyword.")
        else:
            try:
                from pathlib import PureWindowsPath
                p = PureWindowsPath(directory)
                drive = p.drive[0].upper()
                directory_path = f"/host/{drive}/" + "/".join(p.parts[1:])

                async def search_documents():
                    async with streamablehttp_client("http://mcp-server:8000/mcp") as (
                        read_stream, write_stream, _
                    ):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            return await session.call_tool(
                                "search_documents",
                                {
                                    "directory": directory_path,
                                    "query": query,
                                    "max_results": int(max_results)
                                }
                            )

                result = asyncio.run(search_documents())
                raw_result = result.content[0].text
                search_results = json.loads(raw_result)

                folders_searched = search_results.get("folders_searched", 0)
                files_searched = search_results.get("files_searched", 0)
                results = search_results.get("results", [])

                st.success("Search completed.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📁 Folders Searched", folders_searched)
                with col2:
                    st.metric("📄 Files Searched", files_searched)
                with col3:
                    st.metric("🔎 Matches Found", len(results))

                st.write("### Search Results")
                if not results:
                    st.info("No matching files or content were found.")

                for item in results:
                    file_name = item.get("file_name", "Unknown file")
                    file_path = item.get("file", "")
                    match_type = item.get("match_type", "Unknown")
                    context = item.get("context", "")

                    with st.expander(f"📄 {file_name}", expanded=True):
                        st.markdown(f"**Path:** `{file_path}`")
                        st.markdown(f"**Match type:** `{match_type}`")
                        st.markdown(f"**Context:**\n```\n{context}\n```")

            except Exception as e:
                st.error(f"Search failed: {e}")

    # ============================================================
    # RAG SECTION
    # ============================================================
    st.divider()
    st.subheader("🤖 RAG")

    rag_directory = st.text_input("RAG Directory", placeholder=r"D:\All Papers")
    rag_api_key = st.text_input("🔑 OpenAI API Key", type="password", placeholder="sk-...", key="rag_api_key")
    rag_model = st.selectbox("🤖 OpenAI LLM Model", ["gpt-4.1-mini", "gpt-4.1"], index=0, key="rag_model")
    rag_chunk_size = st.number_input("📏 Chunk Size", min_value=200, max_value=5000, value=1000, step=100, key="rag_chunk_size")
    rag_chunk_overlap = st.number_input("↔️ Chunk Overlap", min_value=0, max_value=1000, value=200, step=50, key="rag_chunk_overlap")
    rag_temperature = st.slider("🌡️ Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1, key="rag_temperature")
    rag_question = st.text_area("Question", placeholder="What was the total amount?", height=100, key="rag_question")
    rag_embedding_model = st.selectbox("🧠 OpenAI Embedding Model", ["text-embedding-3-small", "text-embedding-3-large"], index=0, key="rag_embedding_model")


    if st.button("🤖 Run RAG", key="run_rag", use_container_width=True):
    
        if not rag_directory or not rag_api_key or not rag_question:
            st.warning("Please fill in directory, API key, and question.")
    
        else:
    
            try:
    
                # Convert Windows path to Docker path
                from pathlib import PureWindowsPath
    
                p = PureWindowsPath(rag_directory)
    
                drive = p.drive[0].upper()
    
                rag_dir_path = (
                    f"/host/{drive}/"
                    + "/".join(p.parts[1:])
                )
    
                # ------------------------------------------------
                # RUN COMPLETE RAG PIPELINE THROUGH MCP
                # ------------------------------------------------
    
                with st.spinner("🤖 Running RAG through MCP..."):
    
                    rag_result = asyncio.run(
                        call_mcp_rag(
                            directory=rag_dir_path,
                            question=rag_question,
                            api_key=rag_api_key,
                            embedding_model=rag_embedding_model,
                            model_name=rag_model,
                            chunk_size=rag_chunk_size,
                            chunk_overlap=rag_chunk_overlap,
                            temperature=rag_temperature,
                            n_results=5
                        )
                    )
    
                # ------------------------------------------------
                # HANDLE ERROR RETURNED BY RAG SERVER
                # ------------------------------------------------
    
                if "error" in rag_result:
    
                    st.error(
                        f"RAG error: {rag_result['error']}"
                    )
    
                else:
    
                    # ------------------------------------------------
                    # DISPLAY RESULTS
                    # ------------------------------------------------
    
                    st.success(
                        f"RAG completed. "
                        f"Found {rag_result.get('documents_found', 0)} documents."
                    )
    
                    col1, col2, col3, col4 = st.columns(4)
    
                    with col1:
                        st.metric(
                            "📄 Documents",
                            rag_result.get("documents_found", 0)
                        )
    
                    with col2:
                        st.metric(
                            "✂️ Chunks",
                            rag_result.get("chunks_created", 0)
                        )
    
                    with col3:
                        st.metric(
                            "🧠 Chunks Added",
                            rag_result.get("chunks_added", 0)
                        )
    
                    with col4:
                        st.metric(
                            "🔎 Chunks Retrieved",
                            rag_result.get("chunks_retrieved", 0)
                        )
    
                    st.subheader("💡 Answer")
    
                    st.markdown(
                        rag_result.get(
                            "answer",
                            "No answer returned."
                        )
                    )
    
                    sources = rag_result.get(
                        "sources",
                        []
                    )
    
                    if sources:
    
                        st.subheader("📚 Sources")
    
                        for source in sources:
                            st.write(f"- `{source}`")
    
            except Exception as e:
    
                import traceback
    
                st.error(
                    f"RAG failed: {e}"
                )
    
                st.code(
                    traceback.format_exc()
                )




# ============================================================
# REMOTE MCP MODE
# ============================================================
else:
    st.header("🤖 Remote MCP")
    remote_mcp_url = st.text_input("🌐 MCP Server URL", value="https://mcp.deepwiki.com/mcp", key="remote_mcp_url")
    remote_api_key = st.text_input("🔑 OpenAI API Key", type="password", placeholder="sk-...", key="remote_api_key")
    remote_model = st.selectbox("🤖 OpenAI Model", ["gpt-4.1-mini", "gpt-4.1"], index=0, key="remote_model")
    remote_prompt = st.text_area("❓ Prompt", placeholder="Ask the MCP agent anything...", height=120, key="remote_prompt")

    if st.button("🤖 Run Remote MCP", use_container_width=True, key="run_remote_agent"):
        if not remote_api_key or not remote_prompt:
            st.warning("Please enter your API key and prompt.")
        else:
            try:
                with st.spinner("🤖 Agent is working..."):
                    answer = asyncio.run(
                        run_remote_agent(
                            mcp_url=remote_mcp_url,
                            api_key=remote_api_key,
                            model_name=remote_model,
                            prompt=remote_prompt
                        )
                    )
                st.subheader("💡 Result")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Remote MCP Agent failed: {e}")