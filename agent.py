from typing import TypedDict
from langgraph.graph import StateGraph, END
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url
from dotenv import load_dotenv
import re

load_dotenv()
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)



# -----------------------------
# LLM CHAINS
# -----------------------------

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer."),
    ("human", """
Topic: {topic}

Research:
{research}

Write structured report:
- Intro
- Key Findings (3+ points)
- Conclusion
- Sources
""")
])

writer_chain = writer_prompt | llm | StrOutputParser()


critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict evaluator."),
    ("human", """
Report:
{report}

Give:
Score /10
Strengths
Improvements
Final verdict
""")
])

critic_chain = critic_prompt | llm | StrOutputParser()


# -----------------------------
# STATE
# -----------------------------
class ResearchState(TypedDict):
    topic: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str


# -----------------------------
# NODE 1: SEARCH
# -----------------------------
def search_node(state: ResearchState):
    result = web_search.invoke({"query": state["topic"]})
    return {"search_results": result}


# -----------------------------
# NODE 2: SCRAPE
# -----------------------------
def reader_node(state: ResearchState):
    urls = re.findall(r"https?://\S+", state["search_results"])

    if not urls:
        return {"scraped_content": "No URL found"}

    content = scrape_url.invoke({"url": urls[0]})

    return {"scraped_content": content}


# -----------------------------
# NODE 3: WRITER
# -----------------------------
def writer_node(state: ResearchState):
    report = writer_chain.invoke({
        "topic": state["topic"],
        "research": state["search_results"] + "\n\n" + state["scraped_content"]
    })

    return {"report": report}


# -----------------------------
# NODE 4: CRITIC
# -----------------------------
def critic_node(state: ResearchState):
    feedback = critic_chain.invoke({
        "report": state["report"]
    })

    return {"feedback": feedback}


# -----------------------------
# BUILD GRAPH
# -----------------------------
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("reader", reader_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("search")

    graph.add_edge("search", "reader")
    graph.add_edge("reader", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", END)

    return graph.compile()