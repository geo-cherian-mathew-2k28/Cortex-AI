"""
Cortex AI – Agent Core
The agentic reasoning engine that decides when to retrieve, compute, compare,
or restructure outputs. Uses Groq LLM with tool-calling capabilities.
Created by Geo Cherian Mathew.
"""
import json
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime

from groq import Groq, AsyncGroq

from backend.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, MAX_MEMORY_MESSAGES
from utils.embeddings import VectorStore
from utils.chunker import DocumentChunk
from tools.agent_tools import calculator_tool, table_generator_tool, comparison_tool, csv_export_tool


@dataclass
class ChatMessage:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)


class LexiSenseAgent:
    """
    Autonomous agent that decides how to handle user queries:
    1. Direct retrieval from vector store
    2. Computation via calculator tool
    3. Cross-document comparison
    4. Structured table generation
    5. Contextual follow-ups via memory
    """

    SYSTEM_PROMPT = """You are Cortex AI, an autonomous document intelligence agent created by Geo Cherian Mathew. You ONLY analyze uploaded documents and provide intelligent, accurate answers strictly from them.

## Creator:
- Cortex AI was created and developed by **Geo Cherian Mathew**.
- If anyone asks who made you, who created you, or who built you, respond: "I am Cortex AI, created by **Geo Cherian Mathew** — an enterprise-grade document intelligence platform."

## CRITICAL RULES (MUST FOLLOW):
1. **NEVER answer from general knowledge.** You are NOT a general-purpose AI. You are a document analysis tool.
2. **ONLY use information from the retrieved document chunks** provided in the context below.
3. If the retrieved context says "No relevant information found" or the context does not contain the answer, you MUST respond with:
   "I can only answer questions based on your uploaded documents. I don't have information about that topic in the current documents. Please upload relevant files or ask a question about your existing documents."
4. **Do NOT supplement with external knowledge**, even if you know the answer. Stay strictly within the document scope.
5. The ONLY exception is questions about yourself (Cortex AI) or your creator (Geo Cherian Mathew).

## Your Capabilities (Document-Only):
1. **Semantic Search** – Find relevant information across all uploaded documents
2. **Cross-Document Comparison** – Compare content between different files
3. **Structured Extraction** – Extract data into organized tables
4. **Calculations** – Perform mathematical computations on extracted data
5. **Summarization** – Generate concise summaries of documents or topics

## Response Guidelines:
- Always cite which document and section your answer comes from
- When comparing documents, be specific about similarities and differences
- Format tables using markdown when presenting structured data
- Be concise but thorough – prioritize accuracy over verbosity
- If you perform a calculation, show your work
- Use bullet points and headers for readability

## Tool Usage:
When you need to perform specific actions, indicate them in your reasoning:
- [CALC: expression] – For mathematical calculations
- [TABLE: data description] – When structuring data into tables
- [COMPARE: file_a, file_b] – When comparing documents
- [EXPORT: data] – When preparing data for CSV export

## Important:
- NEVER hallucinate or make up information not present in the documents
- If uncertain, say so explicitly
- Always ground your answers in the retrieved document chunks
- If no documents are uploaded, tell the user to upload documents first"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.memory: List[ChatMessage] = []
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.async_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.document_summaries: Dict[str, str] = {}

    def _build_context(self, query: str, file_filter: Optional[str] = None) -> str:
        """Retrieve relevant chunks and build context for the LLM."""
        results = self.vector_store.hybrid_search(query, file_filter=file_filter)

        if not results:
            return "No relevant information found in the uploaded documents."

        context_parts = []
        for chunk, score in results:
            source = f"[Source: {chunk.filename}"
            if chunk.page_info:
                source += f", {chunk.page_info}"
            source += f", Relevance: {score:.2f}]"
            context_parts.append(f"{source}\n{chunk.content}")

        return "\n\n---\n\n".join(context_parts)

    def _detect_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine which tools/actions are needed."""
        query_lower = query.lower()

        intent = {
            "needs_retrieval": True,
            "needs_calculation": False,
            "needs_comparison": False,
            "needs_table": False,
            "needs_export": False,
            "comparison_files": [],
            "calculation_expr": "",
        }

        # Detect calculation intent
        calc_patterns = [
            r'calculate', r'compute', r'what is \d', r'how much',
            r'total', r'sum of', r'average', r'percentage', r'difference between.*\d',
        ]
        if any(re.search(p, query_lower) for p in calc_patterns):
            intent["needs_calculation"] = True

        # Detect comparison intent
        comp_patterns = [
            r'compare', r'difference', r'versus', r'vs\.?',
            r'file a.*file b', r'between.*and', r'contrast',
        ]
        if any(re.search(p, query_lower) for p in comp_patterns):
            intent["needs_comparison"] = True
            # Try to extract file references
            files = self.vector_store.get_files()
            mentioned = [f for f in files if f.lower() in query_lower]
            intent["comparison_files"] = mentioned[:2]

        # Detect table/structured output intent
        table_patterns = [
            r'table', r'list all', r'extract all', r'create a table',
            r'organize', r'structured', r'tabulate',
        ]
        if any(re.search(p, query_lower) for p in table_patterns):
            intent["needs_table"] = True

        # Detect export intent
        if re.search(r'export|download|csv|save as', query_lower):
            intent["needs_export"] = True

        return intent

    def _build_messages(self, query: str, context: str, intent: Dict) -> List[Dict]:
        """Build the message list for the LLM call."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Add recent memory (last N messages)
        recent = self.memory[-MAX_MEMORY_MESSAGES:]
        for msg in recent:
            messages.append({"role": msg.role, "content": msg.content})

        # Build the user message with context
        user_message_parts = []

        # Add available files info
        files = self.vector_store.get_files()
        if files:
            user_message_parts.append(f"**Available Documents:** {', '.join(files)}")

        # Add retrieved context
        user_message_parts.append(f"\n**Retrieved Context:**\n{context}")

        # Add intent-specific instructions
        if intent["needs_comparison"]:
            user_message_parts.append(
                "\n**Note:** The user wants a comparison. Please provide a structured comparison with "
                "similarities, differences, and conflicting information."
            )
        if intent["needs_table"]:
            user_message_parts.append(
                "\n**Note:** The user wants structured/tabular output. Format your response using markdown tables."
            )
        if intent["needs_calculation"]:
            user_message_parts.append(
                "\n**Note:** The user needs calculations. Show your work and provide precise numbers."
            )

        user_message_parts.append(f"\n**User Question:** {query}")

        messages.append({"role": "user", "content": "\n".join(user_message_parts)})
        return messages

    def _process_tool_calls(self, response_text: str) -> str:
        """Process any tool calls embedded in the response."""
        # Handle [CALC: ...] patterns
        calc_matches = re.findall(r'\[CALC:\s*(.+?)\]', response_text)
        for expr in calc_matches:
            result = calculator_tool(expr)
            response_text = response_text.replace(f"[CALC: {expr}]", f"**🧮 {result}**")

        return response_text

    def chat(self, query: str, file_filter: Optional[str] = None) -> str:
        """
        Process a user query synchronously.
        Returns the agent's response.
        """
        if not self.client:
            return "⚠️ Groq API key not configured. Please set GROQ_API_KEY in your .env file."

        # Save user message
        self.memory.append(ChatMessage(role="user", content=query))

        # Detect intent
        intent = self._detect_intent(query)

        # Build context from retrieval
        context = self._build_context(query, file_filter=file_filter)

        # Handle comparison
        if intent["needs_comparison"] and len(intent["comparison_files"]) >= 2:
            file_a, file_b = intent["comparison_files"][:2]
            chunks_a = self.vector_store.get_file_chunks(file_a)
            chunks_b = self.vector_store.get_file_chunks(file_b)
            text_a = "\n".join(c.content for c in chunks_a[:5])
            text_b = "\n".join(c.content for c in chunks_b[:5])
            comparison_context = comparison_tool(text_a, text_b, file_a, file_b)
            context = comparison_context + "\n\n" + context

        # Build messages
        messages = self._build_messages(query, context, intent)

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            answer = response.choices[0].message.content

            # Process embedded tool calls
            answer = self._process_tool_calls(answer)

            # Save assistant response
            self.memory.append(ChatMessage(role="assistant", content=answer))

            return answer

        except Exception as e:
            error_msg = f"⚠️ Error generating response: {str(e)}"
            self.memory.append(ChatMessage(role="assistant", content=error_msg))
            return error_msg

    async def chat_stream(self, query: str, file_filter: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Process a user query with streaming response.
        Yields chunks of the response as they arrive.
        """
        if not self.async_client:
            yield "⚠️ Groq API key not configured. Please set GROQ_API_KEY in your .env file."
            return

        # Save user message
        self.memory.append(ChatMessage(role="user", content=query))

        # Detect intent
        intent = self._detect_intent(query)

        # Build context
        context = self._build_context(query, file_filter=file_filter)

        # Handle comparison
        if intent["needs_comparison"] and len(intent["comparison_files"]) >= 2:
            file_a, file_b = intent["comparison_files"][:2]
            chunks_a = self.vector_store.get_file_chunks(file_a)
            chunks_b = self.vector_store.get_file_chunks(file_b)
            text_a = "\n".join(c.content for c in chunks_a[:5])
            text_b = "\n".join(c.content for c in chunks_b[:5])
            comparison_context = comparison_tool(text_a, text_b, file_a, file_b)
            context = comparison_context + "\n\n" + context

        messages = self._build_messages(query, context, intent)

        try:
            stream = await self.async_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=True,
            )

            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text

            # Process tool calls in complete response
            processed = self._process_tool_calls(full_response)
            if processed != full_response:
                yield "\n\n" + processed.replace(full_response, "")

            self.memory.append(ChatMessage(role="assistant", content=full_response))

        except Exception as e:
            error_msg = f"\n\n⚠️ Error: {str(e)}"
            self.memory.append(ChatMessage(role="assistant", content=error_msg))
            yield error_msg

    def generate_summary(self, filename: str) -> str:
        """Generate an intelligent summary of a document."""
        chunks = self.vector_store.get_file_chunks(filename)
        if not chunks:
            return "No content found for this file."

        # Use first several chunks for summary
        text = "\n\n".join(c.content for c in chunks[:8])

        summary_prompt = f"""Analyze this document and provide a comprehensive Quick View summary:

## Document: {filename}

{text[:6000]}

---

Provide the following in a well-structured format:

### 📋 Executive Summary
(2-3 sentence overview)

### 🔑 Key Points
(Bullet points of the most important information)

### 👤 Important Entities
(Names, organizations, dates, monetary values found)

### 📊 Document Type & Structure
(What type of document this is and how it's organized)

### ⚠️ Risk Indicators / Notable Items
(Any warnings, deadlines, obligations, or critical items)

Be precise and factual. Only include information actually present in the document."""

        if not self.client:
            return "⚠️ Groq API key not configured."

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a document analysis expert. Provide accurate, well-structured summaries."},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            summary = response.choices[0].message.content
            self.document_summaries[filename] = summary
            return summary
        except Exception as e:
            return f"⚠️ Error generating summary: {str(e)}"

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory = []

    def get_memory_context(self) -> List[Dict]:
        """Get conversation history for display."""
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in self.memory
        ]
