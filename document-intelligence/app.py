import streamlit as st
import asyncio
import json
import chromadb
from pathlib import Path
import fitz
from docx import Document
from openpyxl import load_workbook
from openai import OpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

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

        # ----------------------------------------------------
        # OpenAI model
        # ----------------------------------------------------

        model = ChatOpenAI(
            model=model_name,
            api_key=api_key
        )

        # ----------------------------------------------------
        # Connect to MCP
        # ----------------------------------------------------

        mcp_ctx = streamablehttp_client(
            mcp_url,
            timeout=60,
            sse_read_timeout=120
        )

        read_stream, write_stream, _ = (
            await mcp_ctx.__aenter__()
        )

        # ----------------------------------------------------
        # MCP session
        # ----------------------------------------------------

        session_ctx = ClientSession(
            read_stream,
            write_stream
        )

        session = await session_ctx.__aenter__()

        await session.initialize()

        # ----------------------------------------------------
        # Load MCP tools
        # ----------------------------------------------------

        mcp_tools = await load_mcp_tools(
            session
        )

        # ----------------------------------------------------
        # Create agent
        # ----------------------------------------------------

        agent = create_agent(
            model=model,
            tools=mcp_tools
        )

        # ----------------------------------------------------
        # Run agent
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Get final AI response
        # ----------------------------------------------------

        messages = result.get(
            "messages",
            []
        )

        for message in reversed(messages):

            if getattr(
                message,
                "type",
                None
            ) == "ai":

                return message.content

        return "No AI response returned."

    finally:

        # ----------------------------------------------------
        # Close session
        # ----------------------------------------------------

        if session_ctx is not None:

            try:
                await session_ctx.__aexit__(
                    None,
                    None,
                    None
                )
            except Exception:
                pass

        # ----------------------------------------------------
        # Close MCP connection
        # ----------------------------------------------------

        if mcp_ctx is not None:

            try:
                await mcp_ctx.__aexit__(
                    None,
                    None,
                    None
                )
            except Exception:
                pass
            

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 Document Intelligence")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Configuration")

mcp_mode = st.sidebar.radio(
    "MCP Mode",
    [
        "Local MCP",
        "Remote MCP"
    ]
)


# ============================================================
# LOCAL MCP - DRIVE SEARCH
# ============================================================

if mcp_mode == "Local MCP":

    st.header("🏠 Local MCP")

    st.subheader("🔍 Drive Search")

    st.markdown(
        """
        <div style="
            font-size:18px;
            line-height:1.6;
            margin-bottom:25px;
        ">
        Searches recursively through a user-provided directory, including
        PDF, CSV, Python, TXT, Markdown, and other supported text-based files.
        The search checks both <b>file names</b> and <b>file contents/context</b>
        and returns the <b>file path, match type, and relevant context</b>
        for each keyword match.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # INPUTS
    # ========================================================

    directory = st.text_input(
        "Directory",
        placeholder=r"D:\Bank_Statements"
    )

    query = st.text_input(
        "Keyword",
        placeholder="costco"
    )

    max_results = st.number_input(
        "Maximum Results",
        min_value=1,
        max_value=100,
        value=20
    )


    # ========================================================
    # SEARCH BUTTON
    # ========================================================

    if st.button(
        "🔍 Search",
        use_container_width=True
    ):

        if not directory or not query:

            st.warning(
                "Please enter both a directory and keyword."
            )

        else:

            try:

                # ====================================================
                # MCP SEARCH FUNCTION
                # ====================================================
                
                # Mount the whole Windows drives into Docker
                from pathlib import PureWindowsPath
                p = PureWindowsPath(directory)
                drive = p.drive[0].upper()
                directory = f"/host/{drive}/" + "/".join(p.parts[1:])

                async def search_documents():

                    async with streamablehttp_client(
                        "http://mcp-server:8000/mcp"
                    ) as (
                        read_stream,
                        write_stream,
                        _
                    ):

                        async with ClientSession(
                            read_stream,
                            write_stream
                        ) as session:

                            await session.initialize()

                            result = await session.call_tool(
                                "search_documents",
                                {
                                    "directory": directory,
                                    "query": query,
                                    "max_results": int(max_results)
                                }
                            )

                            return result


                # ====================================================
                # RUN MCP SEARCH
                # ====================================================

                result = asyncio.run(
                    search_documents()
                )


                # ====================================================
                # PARSE MCP RESPONSE
                # ====================================================

                raw_result = result.content[0].text

                search_results = json.loads(
                    raw_result
                )


                # ====================================================
                # SEARCH STATISTICS
                # ====================================================

                folders_searched = search_results.get(
                    "folders_searched",
                    0
                )

                files_searched = search_results.get(
                    "files_searched",
                    0
                )

                results = search_results.get(
                    "results",
                    []
                )


                # ====================================================
                # SUCCESS MESSAGE
                # ====================================================

                st.success(
                    "Search completed."
                )


                # ====================================================
                # SEARCH STATISTICS
                # ====================================================

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "📁 Folders Searched",
                        folders_searched
                    )

                with col2:

                    st.metric(
                        "📄 Files Searched",
                        files_searched
                    )

                with col3:

                    st.metric(
                        "🔎 Matches Found",
                        len(results)
                    )


                # ====================================================
                # SEARCH RESULTS
                # ====================================================

                st.write(
                    f"### Search Results"
                )


                if not results:

                    st.info(
                        "No matching files or content were found."
                    )


                # ====================================================
                # DISPLAY RESULTS
                # ====================================================

                for i, item in enumerate(
                    results,
                    start=1
                ):

                    file_name = item.get(
                        "file_name",
                        "Unknown file"
                    )

                    file_path = item.get(
                        "file",
                        ""
                    )

                    match_type = item.get(
                        "match_type",
                        "Unknown"
                    )

                    context = item.get(
                        "context",
                        ""
                    )


                    with st.expander(
                        f"📄 {file_name}",
                        expanded=True
                    ):

                        st.markdown(
                            f"**Path:** `{file_path}`"
                        )

                        st.markdown(
                            f"**Match type:** `{match_type}`"
                        )

                        st.markdown(
                            "**Context:**"
                        )

                        st.markdown(
                            f"""
                            <div style="
                                font-size:16px;
                                line-height:1.6;
                                padding:10px;
                                background-color:#f5f5f5;
                                border-radius:5px;
                                margin-top:5px;
                                margin-bottom:10px;
                            ">
                            {context}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


            except Exception as e:

                st.error(
                    f"Search failed: {e}"
                )


    #
    # ============================================================
    # READ DOCUMENTS FOR RAG
    # ============================================================
    
    TEXT_EXTENSIONS = {
        ".txt", ".md", ".markdown", ".log",
        ".py", ".pyw", ".js", ".ts",
        ".java", ".c", ".cpp", ".h", ".hpp",
        ".r", ".sql",
        ".json", ".xml", ".html", ".htm",
        ".csv", ".tsv",
        ".ipynb",
    }
    
    
    def read_documents_for_rag(directory):
    
        root = Path(directory)
    
        if not root.exists():
            raise ValueError(
                f"Directory does not exist: {directory}"
            )
    
        documents = []
    
        for path in root.rglob("*"):
    
            if not path.is_file():
                continue
    
            extension = path.suffix.lower()
    
            # ----------------------------------------------------
            # TEXT FILES
            # ----------------------------------------------------
    
            if extension in TEXT_EXTENSIONS:
    
                try:
    
                    with open(
                        path,
                        "r",
                        encoding="utf-8",
                        errors="ignore"
                    ) as f:
    
                        text = f.read()
    
                except Exception:
                    continue
    
            # ----------------------------------------------------
            # PDF
            # ----------------------------------------------------
    
            elif extension == ".pdf":
    
                try:
    
                    text_parts = []
    
                    with fitz.open(path) as pdf:
    
                        for page_number, page in enumerate(
                            pdf,
                            start=1
                        ):
    
                            page_text = page.get_text()
    
                            if page_text.strip():
    
                                text_parts.append(
                                    f"[PAGE {page_number}]\n"
                                    f"{page_text}"
                                )
    
                    text = "\n\n".join(
                        text_parts
                    )
    
                except Exception:
                    continue
    
            # ----------------------------------------------------
            # DOCX
            # ----------------------------------------------------
    
            elif extension == ".docx":
    
                try:
    
                    document = Document(path)
    
                    text_parts = []
    
                    for paragraph in document.paragraphs:
    
                        if paragraph.text.strip():
    
                            text_parts.append(
                                paragraph.text
                            )
    
                    text = "\n".join(
                        text_parts
                    )
    
                except Exception:
                    continue
    
            # ----------------------------------------------------
            # EXCEL
            # ----------------------------------------------------
    
            elif extension in {".xlsx", ".xlsm"}:
    
                try:
    
                    workbook = load_workbook(
                        path,
                        read_only=True,
                        data_only=True
                    )
    
                    text_parts = []
    
                    for worksheet in workbook.worksheets:
    
                        text_parts.append(
                            f"[SHEET: {worksheet.title}]"
                        )
    
                        for row in worksheet.iter_rows(
                            values_only=True
                        ):
    
                            values = [
                                str(value)
                                for value in row
                                if value is not None
                            ]
    
                            if values:
    
                                text_parts.append(
                                    " | ".join(values)
                                )
    
                    workbook.close()
    
                    text = "\n".join(
                        text_parts
                    )
    
                except Exception:
                    continue
    
            else:
    
                continue
    
            # ----------------------------------------------------
            # SAVE DOCUMENT
            # ----------------------------------------------------
    
            if text and text.strip():
    
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": str(path),
                        "file_name": path.name,
                        "file_type": extension
                    }
                })
    
        return documents
    
    #
    # ============================================================
    # CREATE CHUNKS FOR RAG
    # ============================================================
    
    def create_chunks(
        documents,
        chunk_size=1000,
        chunk_overlap=200
    ):
        """
        Split documents into overlapping text chunks.
    
        Input:
            documents = [
                {
                    "text": "...",
                    "metadata": {...}
                }
            ]
    
        Output:
            [
                {
                    "text": "...",
                    "metadata": {...}
                }
            ]
        """
    
        chunks = []
    
        for document in documents:
    
            text = document["text"]
            metadata = document["metadata"]
    
            # Clean whitespace
            text = text.strip()
    
            if not text:
                continue
    
            start = 0
    
            while start < len(text):
    
                end = start + chunk_size
    
                chunk_text = text[start:end].strip()
    
                if chunk_text:
    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": metadata.copy()
                    })
    
                # Move forward while keeping overlap
                start += chunk_size - chunk_overlap
    
        return chunks

    # ============================================================
    # ADD CHUNKS TO CHROMA
    # ============================================================
    
    def add_chunks(
        chunks,
        api_key
    ):
    
        if not chunks:
            return
    
        # --------------------------------------------------------
        # OpenAI client
        # --------------------------------------------------------
    
        openai_client = OpenAI(
            api_key=api_key
        )
    
        # --------------------------------------------------------
        # Chroma
        # --------------------------------------------------------
    
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )
    
        collection = chroma_client.get_or_create_collection(
            name="documents"
        )
    
        # --------------------------------------------------------
        # Prepare data
        # --------------------------------------------------------
    
        texts = []
        metadatas = []
        ids = []
    
        for i, chunk in enumerate(chunks):
    
            texts.append(
                chunk["text"]
            )
    
            metadatas.append(
                chunk["metadata"]
            )
    
            ids.append(
                f"rag_chunk_{i}"
            )
    
        # --------------------------------------------------------
        # OpenAI embeddings
        # --------------------------------------------------------
    
        response = openai_client.embeddings.create(
            model=rag_embedding_model,
            input=texts
        )
    
        embeddings = [
            item.embedding
            for item in response.data
        ]
    
        # --------------------------------------------------------
        # Store in Chroma
        # --------------------------------------------------------
    
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
    
        print(
            f"Added {len(chunks)} chunks to Chroma."
        )


    #
    # ============================================================
    # LOCAL MCP - RAG
    # ============================================================
    
    st.divider()
    
    st.subheader("🤖 RAG")
    
    st.markdown(
        """
        <div style="
            font-size:18px;
            line-height:1.6;
            margin-bottom:25px;
        ">
        Enter a directory and ask a question. The application will
        index the documents in that directory, retrieve the most
        relevant content using semantic similarity, and use an
        OpenAI LLM to generate an answer with sources.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
    # ============================================================
    # DIRECTORY
    # ============================================================
    
    rag_directory = st.text_input(
        "RAG Directory",
        placeholder=r"D:\All Papers"
    )
    
    #
    # ============================================================
    # RAG CONFIGURATION
    # ============================================================
    
    # ------------------------------------------------------------
    # OPENAI API KEY
    # ------------------------------------------------------------
    
    rag_api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="rag_api_key"
    )
    
    
    # ------------------------------------------------------------
    # OPENAI LLM MODEL
    # ------------------------------------------------------------
    
    rag_model = st.selectbox(
        "🤖 OpenAI LLM Model",
        [
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-5-mini",
            "gpt-5.5",
            "gpt-5.6",

        ],
        index=0,
        key="rag_model"
    )
    
    
    # ------------------------------------------------------------
    # CHUNK SIZE
    # ------------------------------------------------------------
    
    rag_chunk_size = st.number_input(
        "📏 Chunk Size",
        min_value=200,
        max_value=5000,
        value=1000,
        step=100,
        key="rag_chunk_size",
        help="Number of characters in each document chunk."
    )
    
    
    # ------------------------------------------------------------
    # CHUNK OVERLAP
    # ------------------------------------------------------------
    
    rag_chunk_overlap = st.number_input(
        "↔️ Chunk Overlap",
        min_value=0,
        max_value=1000,
        value=200,
        step=50,
        key="rag_chunk_overlap",
        help="Number of characters shared between consecutive chunks."
    )
    
    # ------------------------------------------------------------
    # TEMPERATURE
    # ------------------------------------------------------------
    
    rag_temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        key="rag_temperature",
        help="Controls randomness of the LLM response."
    )    

    
    # ============================================================
    # QUESTION
    # ============================================================
    
    rag_question = st.text_area(
        "Question",
        placeholder="What was the total amount of the utility bill?",
        height=100,
        key="rag_question"
    )
    

    #
    # ============================================================
    # CHROMA
    # ============================================================
    
    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )
    
    collection = chroma_client.get_or_create_collection(
        name="documents"
    )
    

    # ============================================================
    # OPENAI EMBEDDINGS
    # ============================================================


    rag_embedding_model = st.selectbox(
        "🧠 OpenAI Embedding Model",
        [
            "text-embedding-3-small",
            "text-embedding-3-large"
        ],
        index=0,
        key="rag_embedding_model"
    )
        
    def create_embedding(
        text,
        api_key
    ):
    
        client = OpenAI(
            api_key=api_key
        )
    
        response = client.embeddings.create(
            model=rag_embedding_model,
            input=[text]
        )
    
        return response.data[0].embedding
    
    
    # ============================================================
    # CHROMA SEARCH
    # ============================================================
    
    def retrieve_documents(
        question,
        api_key,
        n_results=5
    ):
    
        embedding = create_embedding(
            question,
            api_key
        )
    
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
    
        return results
    
    # ============================================================
    # OPENAI RAG ANSWER
    # ============================================================
    
    def generate_rag_answer(
        question,
        documents,
        metadatas,
        api_key
    ):
    
        client = OpenAI(
            api_key=api_key
        )
    
        # --------------------------------------------------------
        # Build context
        # --------------------------------------------------------
    
        context_parts = []
    
        for i, (document, metadata) in enumerate(
            zip(documents, metadatas),
            start=1
        ):
    
            source = metadata.get(
                "source",
                "Unknown source"
            )
    
            context_parts.append(
                f"""
    SOURCE {i}:
    {source}
    
    CONTENT:
    {document}
    """
            )
    
        context = "\n\n".join(
            context_parts
        )
    
        # --------------------------------------------------------
        # Prompt
        # --------------------------------------------------------
    
        prompt = f"""
    You are a document question-answering assistant.
    
    Answer the user's question using ONLY the information
    provided in the document context below.
    
    If the answer cannot be found in the documents,
    say that the information was not found.
    
    Always identify the source document when possible.
    
    USER QUESTION:
    {question}
    
    DOCUMENT CONTEXT:
    {context}
    """
    
        # --------------------------------------------------------
        # OpenAI LLM
        # --------------------------------------------------------
    
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using retrieved "
                        "document context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
        return response.choices[0].message.content



    
    # ============================================================
    # RUN RAG
    # ============================================================
    
    if st.button(
        "🤖 Run RAG",
        key="run_rag",
        use_container_width=True
    ):
    
        if not rag_directory:
    
            st.warning(
                "Please enter a directory."
            )
    
        elif not rag_api_key:
    
            st.warning(
                "Please enter your OpenAI API key."
            )
    
        elif not rag_question:
    
            st.warning(
                "Please enter a question."
            )
    
        else:
    
            try:
    
                # =================================================
                # STEP 1 — READ DIRECTORY
                # =================================================
    
                # Mount the whole Windows drives into Docker
                from pathlib import PureWindowsPath
                p = PureWindowsPath(rag_directory)
                drive = p.drive[0].upper()
                rag_directory = f"/host/{drive}/" + "/".join(p.parts[1:])

                with st.spinner(
                    "📂 Reading documents..."
                ):
    
                    documents = read_documents_for_rag(
                        rag_directory
                    )
    
    
                if not documents:
    
                    st.warning(
                        "No supported documents were found."
                    )
    
                    st.stop()
    
    
                st.success(
                    f"Found {len(documents)} documents."
                )
    
    
                # =================================================
                # STEP 2 — CHUNK
                # =================================================
    
                with st.spinner(
                    "✂️ Creating document chunks..."
                ):
    
                    chunks = create_chunks(
                        documents
                    )
    
    
                st.info(
                    f"Created {len(chunks)} chunks."
                )
    
    
                # =================================================
                # STEP 3 — CREATE EMBEDDINGS
                # =================================================
    
                with st.spinner(
                    "🧠 Creating OpenAI embeddings..."
                ):
    
                    add_chunks(
                        chunks,
                        api_key=rag_api_key
                    )
    
    
                # =================================================
                # STEP 4 — SEMANTIC SEARCH
                # =================================================
    
                with st.spinner(
                    "🔎 Finding relevant information..."
                ):
    
                    retrieved = retrieve_documents(
                        question=rag_question,
                        api_key=rag_api_key,
                        n_results=5
                    )
    
    
                retrieved_documents = retrieved[
                    "documents"
                ][0]
    
                retrieved_metadata = retrieved[
                    "metadatas"
                ][0]
    
    
                if not retrieved_documents:
    
                    st.warning(
                        "No relevant information was found."
                    )
    
                    st.stop()
    
    
                # =================================================
                # STEP 5 — LLM
                # =================================================
    
                with st.spinner(
                    "🤖 Generating answer..."
                ):
    
                    answer = generate_rag_answer(
                        question=rag_question,
                        documents=retrieved_documents,
                        metadatas=retrieved_metadata,
                        api_key=rag_api_key
                    )
    
    
                # =================================================
                # ANSWER
                # =================================================
    
                st.subheader("💡 Answer")
    
                st.markdown(
                    answer
                )
    
    
                # =================================================
                # SOURCES
                # =================================================
    
                st.subheader("📚 Sources")
    
                for i, (
                    document,
                    metadata
                ) in enumerate(
                    zip(
                        retrieved_documents,
                        retrieved_metadata
                    ),
                    start=1
                ):
    
                    source = metadata.get(
                        "source",
                        "Unknown"
                    )
    
                    with st.expander(
                        f"📄 {i}. {source}"
                    ):
    
                        st.markdown(
                            f"**Source:** `{source}`"
                        )
    
                        st.text(
                            document
                        )
    
    
            except Exception as e:
    
                st.error(
                    f"RAG failed: {e}"
                )


# ============================================================
# REMOTE MCP
# ============================================================

else:

    st.header("🤖 Remote MCP")

    st.markdown(
        """
        <div style="
            font-size:18px;
            line-height:1.6;
            margin-bottom:25px;
        ">
        Connect to a Remote MCP server and use its tools through
        an OpenAI-powered agent.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # MCP SERVER URL
    # ========================================================

    remote_mcp_url = st.text_input(
        "🌐 MCP Server URL",
        value="https://mcp.deepwiki.com/mcp",
        key="remote_mcp_url"
    )


    # ========================================================
    # OPENAI API KEY
    # ========================================================

    remote_api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="remote_api_key"
    )


    # ========================================================
    # OPENAI MODEL
    # ========================================================

    remote_model = st.selectbox(
        "🤖 OpenAI Model",
        [
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-5-mini",
            "gpt-5.5",
            "gpt-5.6"
        ],
        index=0,
        key="remote_model"
    )


    # ========================================================
    # CONNECT
    # ========================================================

    if st.button(
        "🔌 Connect",
        use_container_width=True,
        key="remote_connect"
    ):

        try:

            async def get_remote_tools():

                async with streamablehttp_client(
                    remote_mcp_url,
                    timeout=30,
                    sse_read_timeout=120
                ) as (
                    read_stream,
                    write_stream,
                    _
                ):

                    async with ClientSession(
                        read_stream,
                        write_stream
                    ) as session:

                        await session.initialize()

                        tools = await session.list_tools()

                        return [
                            {
                                "name": tool.name,
                                "description": (
                                    tool.description or ""
                                )
                            }
                            for tool in tools.tools
                        ]


            with st.spinner(
                "🔌 Connecting to MCP server..."
            ):

                tools = asyncio.run(
                    get_remote_tools()
                )


            st.session_state[
                "remote_tools"
            ] = tools

            st.session_state[
                "remote_connected"
            ] = True

            st.success(
                "✅ Connected successfully."
            )


        except Exception as e:

            st.session_state[
                "remote_connected"
            ] = False

            st.error(
                f"Connection failed: {e}"
            )


    # ========================================================
    # SHOW MCP TOOLS
    # ========================================================

    if st.session_state.get(
        "remote_connected",
        False
    ):

        tools = st.session_state.get(
            "remote_tools",
            []
        )

        st.subheader(
            f"🛠️ MCP Tools ({len(tools)})"
        )

        for tool in tools:

            with st.expander(
                f"🔧 {tool['name']}"
            ):

                st.write(
                    tool["description"]
                    or "No description available."
                )


    # ========================================================
    # PROMPT
    # ========================================================

    remote_prompt = st.text_area(
        "❓ Prompt",
        placeholder=(
            "Ask the MCP agent anything..."
        ),
        height=120,
        key="remote_prompt"
    )


    # ========================================================
    # RUN REMOTE MCP AGENT
    # ========================================================

    if st.button(
        "🤖 Run Remote MCP",
        use_container_width=True,
        key="run_remote_agent"
    ):

        if not remote_api_key:

            st.warning(
                "Please enter your OpenAI API key."
            )

        elif not remote_prompt:

            st.warning(
                "Please enter a prompt."
            )

        elif not st.session_state.get(
            "remote_connected",
            False
        ):

            st.warning(
                "Please connect to the MCP server first."
            )

        else:

            try:

                with st.spinner(
                    "🤖 Agent is working..."
                ):

                    answer = asyncio.run(
                        run_remote_agent(
                            mcp_url=remote_mcp_url,
                            api_key=remote_api_key,
                            model_name=remote_model,
                            prompt=remote_prompt
                        )
                    )


                # =================================================
                # DISPLAY FINAL RESULT ONLY
                # =================================================

                st.subheader(
                    "💡 Result"
                )

                st.markdown(
                    answer
                )


            except Exception as e:

                st.error(
                    f"Remote MCP Agent failed: {e}"
                )