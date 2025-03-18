import logging
import streamlit as st
import json
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from Agents import get_agents
from VectorStore_Retrieve import get_retriever

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Workflow:
    """
    Handles the workflow for processing retrieval-augmented generation (RAG).
    - Retrieves documents
    - Grades them for relevance
    - Generates an answer
    - Checks hallucination
    - Uses web search if needed
    - Summarizes if asked
    """
    # Initialize the workflow with the model
    def __init__(self, model: str = "meta-llama/llama-3.2-1b-instruct:free"):
        
        (
            self.retrieval_grader,
            self.summary_chain,  
            self.rag_chain,
            self.hallucination_grader,
            self.answer_grader,
            self.web_search_tool,
            self.writing_feedback_agent,  # Added writing feedback agent

            # self.rewrite_query_tool,
        ) = get_agents(model)
        
        self.retriever = get_retriever()

    def build(self):
        """
        Creates the workflow state machine for RAG processing.
        """
        workflow = StateGraph(self.GraphState)

        # Define Workflow Nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("websearch", self.web_search)
        workflow.add_node("summary", self.generate_summary)
        workflow.add_node("writing_feedback", self.writing_feedback)  
  
        # Define Workflow Edges
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("websearch", "generate")
        workflow.add_edge("generate", END)
        workflow.add_edge("summary", END)
        workflow.add_edge("writing_feedback", END)  
 
 
        
        # Add conditional edges for document grading
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        # Add conditional edges for grading generation
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_with_docs_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
            },
        )

       #Add conditional entry point
        workflow.set_conditional_entry_point(
            self.route_chain,  
            {
                "summary": "summary",
                "rag": "retrieve",
                "writing_feedback": "writing_feedback",  # Added route writing-related queries to feedback


            },
        )

        return workflow.compile()

    class GraphState(TypedDict):
        """
        Represents the state of the workflow.

        Attributes:
            question: The user's question.
            generation: The LLM-generated answer.
            web_search: Whether web search is needed.
            documents: List of retrieved documents.
            retries: Number of attempts for generation.
            max_retries: Maximum number of retries allowed.
        """
        question: str
        generation: str
        web_search: str
        documents: List[str]
        retries: int
        max_retries: int

    #Initialize RAG function
    def intialize_rag(self, state):
        logger.info("Initializing RAG system...")
        self.retriever = get_retriever()
        logger.info("RAG system initialized!")
        return state
    
    #Retrieving documentsfrom Vector DB
    def retrieve(self, state):
        logger.info("--- Retrieving Documents ---")
        question = state["question"]
        # Retrieval
        documents = self.retriever.invoke(question)
        return {
            "documents": documents,
            "question": question,
            "retries": state["retries"],
        }
        
    #Grade docs relevant such that it decides if we need to do web search or not
    def grade_documents(self, state):
        logger.info("--- Grading Retrieved Documents ---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = "No"

        for doc in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": doc.page_content})
            grade = score["score"]
            if grade.lower() == "yes":
                filtered_docs.append(doc)
            else:
                web_search = "Yes"

        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
            "retries": state["retries"],
        }
    #Generating Answer function
    def generate(self, state):
        logger.info("--- Generating Answer ---")
        question = state["question"]
        documents = state["documents"]

        if not documents:
            return {"generation": " No relevant documents found to answer this question."}

        generation = self.rag_chain.invoke({"context": documents, "question": question})

        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "retries": state["retries"],
        }
    #Generating Summary function    
    def generate_summary(self, state):
        """
        Generates a summary using an optimized map-reduce strategy with concurrency.
        """
        logger.info("--- Generating Optimized Summary ---")
        documents = state["documents"]

        # Check Input Type
        print(f"Summary Input Type: {type(documents)}")

        if not isinstance(documents, list) or not all(isinstance(doc, Document) for doc in documents):
            return {"generation": "Invalid document format."}

        try:
            # # Limit Number of Chunks to Process (top 20 chunks)
            # max_chunks = 20
            # selected_docs = documents[:max_chunks]

            print(f"Selected {len(documents)} documents for summary.")

            # Use 'map_reduce' with Parallelism Enabled
            summary_result = self.summary_chain.invoke(
                {
                    "input_documents": documents,
                    "config": {"max_concurrency": 8},  # Adjust concurrency based on machine's capacity
                }
            )

            # Return Summary
            print(f"Summary Generated: {summary_result}")
            return {"generation": summary_result.get("output_text", "No summary generated.")}

        except Exception as e:
            print(f"Error generating summary: {e}")
            logger.error(f"Error generating summary: {e}")
            return {"generation": f"Error generating summary: {str(e)}"}

    
    def writing_feedback(self, state):
        """
        Generates personalized feedback for writing.
        """
        logger.info("--- Providing Writing Feedback ---")
        retrieved_docs = "\n".join([doc.page_content for doc in state["documents"]]) if state["documents"] else ""

        if not retrieved_docs:
            return {"generation": "No documents found for writing feedback."}
   
        feedback = self.writing_feedback_agent.invoke({"retrieved_docs": retrieved_docs})

        return {
            "documents": state["documents"],
            "question": state["question"],
            "generation": feedback,
        }
        
        
    #Web search function
    def web_search(self, state):
        print("Performing web search")
        logger.info("--- Performing Web Search ---")
        question = state["question"]
        documents = state["documents"]

        search_results = self.web_search_tool.invoke({"query": question})
        web_results = Document(page_content=search_results)
        print(f"Web search results: {web_results}")
        documents.append(web_results)

        return {
            "documents": documents,
            "question": question,
            "retries": state["retries"],
        }
        
    #Route Between Summary, Writing feedback and RAG    
    def route_chain(self, state):
        """
        Routes the workflow execution based only on the question:
        - If the question is include wordings of 'summary', route to 'summary'.
        - If the question is include wordings of 'grammar', 'writing feedback', 'check my writing', route to 'writing_feedback'.
        - Otherwise, route to 'rag'.
        """
        logger.info("---ROUTE CHAIN---")
        question = state["question"].strip().lower()

        # Quick Debug Logging
        print(f"Routing chain: Question='{question}'")

        if "summary" in question or "summarize" in question or "what is the objective" in question:
            logger.info("---ROUTE QUESTION TO SUMMARY---")
            print("ROUTE QUESTION TO SUMMARY")
            return "summary"
        elif "grammar" in question or "writing feedback" in question or "check my writing" in question:
            logger.info("---ROUTE QUESTION TO WRITING FEEDBACK---")
            return "writing_feedback"

        else:
            logger.info("---ROUTE QUESTION TO RAG---")
            print("ROUTE QUESTION TO RAG")
            return "rag"
    
    #Grade generation function
    def grade_generation_with_docs_and_question(self, state):
        logger.info("--- Checking Answer Quality ---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        retries = state.get("retries", 0)
        max_retries = state.get("max_retries", 3)
        # MAX_RETRIES = 3

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score["score"]
        
        if grade == "yes":  # No hallucination
            answer_score = self.answer_grader.invoke({"question": question, "generation": generation})
            if answer_score["score"] == "yes":  # If answer is useful
                return "useful"
            else:  # If answer is not useful
                if retries >= max_retries:
                    logger.warning(" Max retries reached. Stopping reattempts.")
                    return "useful"  # Prevent looping, return the last attempt
                
                return "not useful"
        else:  # If hallucination detected
            logger.info("--- Generation is NOT grounded. Re-attempting... ---")
            if retries >= max_retries:
                logger.warning(" Max retries reached. Stopping reattempts.")
                return "useful"
            # Modify the query to request better grounding
            refined_query = question + " (Ensure the response is based only on provided documents.)"
            state["question"] = refined_query  # Update the query to guide LLM towards fact-based answers

            # Clear hallucinated response before retrying
            state["generation"] = f"Retrying due to detected hallucination... (Attempt {retries + 1})"
            state["retries"] = retries + 1  # Increment retry counter
            
            return "not supported"

    #Retireval decider
    def decide_to_generate(self, state):
        logger.info("--- Evaluating Retrieved Documents ---")
        web_search = state["web_search"]
        retries = state.get("retries", 0)
        MAX_RETRIES = 3

        if web_search == "Yes" and retries < MAX_RETRIES:
            logger.info("---Documents are irrelevant. Performing web search---")
            return "websearch"
        else:
            logger.info("---Documents are relevant. Generating answer---")  
            return "generate"
        

    
    
    

