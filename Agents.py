import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama import ChatOllama  
from langchain_google_community import GoogleSearchAPIWrapper 
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains import LLMChain  # Ensure this import exists


load_dotenv()


GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

#Prompt from https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
# Calling LLM and using prompt https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_local.ipynb
PROMPT_TEMPLATES = {
    "retrieval_grader": """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document to a user question. If the document contains any information that would answer the user question, grade it as relevant.
    If the users ask about the content of the document, grade it as relevant also.Relevant then return 'yes'Return a binary 'yes' or 'no' score to indicate whether the document is relevant to the question.The binary score should be in JSON format as 'score': "<yes/no>" with no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    The retrieved document: \n\n {document} \n\n
    User question: {question} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    
    "rag_chain": """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. Use only these following retrieved context to answer the question. If you don't know the answer, just say you don't know politely.
    Keep the answer concise, relevant to the question and documents. Keep the answer around 100-200 words. Cite the source of the answer if possible. Use the file name from user_iploaded_files if mentioned.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    
    "hallucination_grader": """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grading assistant that assesses whether the answer is supported by a set of facts. Return a binary 'yes' or 'no' score to indicate whether the answer is supported by a set of facts. 
    The binary score should be in JSON format as 'score': "<yes/no>" with no preamble or explanation. 
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    
    "answer_grader": """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grading assistant assessing whether an answer addresses/ resolves a question.If the user asks about the content of the document, check if the retrieved context can resolve the question.Return a binary score 'yes' or 'no' to indicate whether the answer can resolve a question. 'Yes' means the answer rsolves the question.
    The binary score should be in JSON format as 'score': "<yes/no>" with no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|> 
    Now is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    
    # "rewrite_query": """
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # You are an AI assistant rewriting the user question to improve search results.
    # Rewrite the user question to improve search results and also useful for the question.    
    # The user previously searched for:
    # "{question}"
    # but the retrieved answer was not useful:
    # "{previous_answer}"
    # Rewrite the search query to improve relevance while maintaining its original intent.
    # Return only the refined query without explanations.
    # <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    # """,
    
    "map_template": """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are now a summarization assistant. The following is part of a document that needs to be summarized. Your task is to analyze the following paragph excerpts and extract key details, main ideas, and important insights.
    The summary should capture the core message of each document, identify recurring themes, facts, and patterns, and maintain factual accuracy.:
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the document:
    \n ------- \n
    {docs}
    \n ------- \n
    Now provide a structured and consise summary of this paragraph.
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    
    "reduce_template": """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert summarization assistant. The following is a set of summaries extracted from multiple documents.Your task is to analyze these summaries and combine them into a final consolidated summary.
    Your final summary should merge overlapping ideas, maintain clarity, coherence, and logical flow, resolve any conflicting information, and keep the summary concise, factual, and well-structured.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the summaries:
    \n ------- \n
    {docs}
    \n ------- \n
    Now provide a final consolidated summary of the above summaries.
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    
    "writing_feedback": """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an AI writing assistant that provides ersonalized feedback on writing.Your goal is to help users improve their writing skills by:
    - Offering changes according to user's question.
    - Provide feedback based on the detected style.
    - Highlight grammar mistakes and suggest corrections.
    - Suggest tone adjustments if needed.
    - Highlight any mistakes or areas for improvement and provide corrected versions, explanations on why the changes improve the writing.  
    Point out the sentences that needs to be changed and offer an improve one. Do not generate the whole passage. If no major errors exist, offer style suggestions to refine the writing further.  Use simple and clear language to ensure the user understands your feedback.  
    Your role is to guide, help the user feel motivated to improve their writing.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the text that needs feedback:  
    \n ------- \n
    {retrieved_docs}
    \n ------- \n
    Please provide the analysis.
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
}

def get_llm(model: str = "meta-llama/llama-3.1-8b-instruct:free", temperature=0):
    """
    Initializes an LLM instance using OpenRouter via LangChain.
    """
    return ChatOpenAI(
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        model_name=model,
        temperature=temperature,
        
    )
# def get_llm(model: str = "llama3.2:3b", format="json", temperature=0,num_ctx: int = 12000): #Context size is 8000
#     """
#     Initializes a ChatOllama LLM instance.
#     """
#     return ChatOllama(model=model,format=format,temperature=temperature,timeout=5000,num_ctx=num_ctx)

def get_map_reduce_chain(model: str):
    """
    Creates a Map-Reduce summarization chain using LangChain components.
    """
    # llm = ChatOllama(model=model, temperature=0, num_ctx=12000)
    llm= get_llm(model)

    # Define map and reduce prompt templates
    map_prompt = PromptTemplate(template=PROMPT_TEMPLATES["map_template"], input_variables=["docs"])
    reduce_prompt = PromptTemplate(template=PROMPT_TEMPLATES["reduce_template"], input_variables=["docs"])

    # Map step: Summarizing individual document chunks
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce step: Combining summaries into a final summary
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes multiple mapped summaries and combines them before reducing
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # ReduceDocumentsChain: Handles document summarization iteratively
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000  # Adjust based on LLM token limit
    )

    # Final Map-Reduce chain
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False
    )
    return map_reduce_chain


def create_agent(model: str, prompt_key: str, output_parser):
    """
    Creates an agent by combining a prompt, LLM, and an output parser.
    """
    input_vars_map = {
        "rag_chain": ["question", "context"],
        "retrieval_grader": ["question", "document"],
        "hallucination_grader": ["documents", "generation"],
        "answer_grader": ["question", "generation"],
        # "rewrite_query": ["question", "previous_answer"],  # Uses previous answer for refining query
    }
    prompt_template = PromptTemplate(
        template=PROMPT_TEMPLATES[prompt_key],
        # Ensure correct input variables for the prompt
        # input_variables = input_vars_map.get(prompt_key, ["context"])
        input_variables=["text"] if prompt_key == "writing_feedback" else ["question", "context"],
    )
    # llm = get_llm(model, format="json" if isinstance(output_parser, JsonOutputParser) else None)
    llm = get_llm(model)

    return prompt_template | llm | output_parser

def form_document_context(documents: list):
    return "\n".join(documents for documents in documents)

def get_agents(model: str, num_ctx: int = 12000):
    """
    Initializes all agents for RAG processing.
    """
    # llm = get_llm(model, num_ctx=num_ctx)
    writing_feedback_agent = create_agent(model, "writing_feedback", StrOutputParser())
    retrieval_grader = create_agent(model, "retrieval_grader", JsonOutputParser())
    rag_chain = create_agent(model, "rag_chain", StrOutputParser())
    summary_chain = get_map_reduce_chain(model)
    
    hallucination_grader = create_agent(model, "hallucination_grader", JsonOutputParser())
    answer_grader = create_agent(model, "answer_grader", JsonOutputParser())
    
    search = GoogleSearchAPIWrapper()
    web_search_tool = Tool(
        name="google_search",
        description="Search relevant information.",
        func=search.run,
    )
    # rewrite_query_tool = create_agent(model, "rewrite_query", StrOutputParser())
    
    
    return (
        retrieval_grader,
        summary_chain,
        rag_chain,
        hallucination_grader,
        answer_grader,
        web_search_tool,
        writing_feedback_agent
        # rewrite_query_tool
    )
    
