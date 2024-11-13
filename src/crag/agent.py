from typing_extensions import TypedDict, List
from langgraph.graph import START, END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
from src.crag.utils import saveGraph
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question to be used as input in LLM chain
        generation: LLM generation response
        search: "yes" or "no" string acting as boolean for whether to invoke web search
        documents: list of documents for in-context learning
        steps: List of steps taken in agent flow
        user_query: original user query, stored here for persistence during consolidation stage
        sub_answers: list of answers to decomposed questions
    """
    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]
    user_query: str
    sub_answers: List[str]
    sub_questions: List[str]


class RagAgentInput(TypedDict):
    agent_name: str
    retriever: any
    query_decompose_llm: any
    grader_llm: any
    rag_llm: any
    web_search_tool: TavilySearchResults


class RagAgent():
    def __init__(self, params: RagAgentInput):
        self.params = params
        # Define LLM chains
        self.query_decompose_chain = self._getQueryDecomposeChain()
        self.grader_chain = self._getGraderChain()
        self.rag_chain = self._getRAGChain()
        
        # Build Agent Graphs
        self.compiled_graph = self._create_crag_graph()
        self.nested_compiled_graph = self._create_nested_crag_graph()
    

    def query(self,question: str) -> str:
        return self.nested_compiled_graph.invoke({"user_query": question, "steps": []})

    def _create_crag_graph(self):
        graph = StateGraph(GraphState)
        # Define the nodes
        graph.add_node("retrieve", lambda state: self._retrieve(state))  # retrieve
        graph.add_node("grade_documents", lambda state: self._grade_documents(state)) # grade documents
        graph.add_node("generate", lambda state: self._generate(state))  # generatae
        graph.add_node("web_search", lambda state: self._web_search(state))  # web search
        # Build graph
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",  #at grade_documents node, invoke decide_to_generate function
            lambda state: self._decide_to_generate(state),
            {
        "search": "web_search", #if "search" is returned, invoke the "web_search" node
        "generate": "generate", #if "generate" is returned, invoke the "generate" node
        },
        )
        graph.add_edge("web_search", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def _create_nested_crag_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("transform_query", lambda state: self._transform_query(state))  # retrieve
        graph.add_node("CRAG_loop",lambda state: self._crag_loop(state))
        graph.add_node("consolidate",lambda state: self._consolidate(state))
        graph.set_entry_point("transform_query")
        graph.add_edge("transform_query", "CRAG_loop")
        graph.add_edge("CRAG_loop", "consolidate")
        graph.add_edge("consolidate", END)
        return graph.compile()
    
    def saveGraphImage(self):
        saveGraph(f'./graphs/{self.params["agent_name"]}_crag',self.compiled_graph)
        saveGraph(f'./graphs/{self.params["agent_name"]}_nested_crag',self.nested_compiled_graph)

    def _retrieve(self,state):
        """
        Retrieve documents
        This is the first Node invoked in the CRAG_graph

        # CRAG_graph is invoked in the CRAG_loop node:
        #response = CRAG_graph.invoke({"question": q, "steps": steps})["generation"]
        #we initialize the state with a sub-question and list of steps

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---Retrieving Documents---")
        """-----------inputs-----------"""
        question = state["question"]
        steps = state["steps"]

        """-----------actions-----------"""
        steps.append("retrieve_documents")
        documents = self.params["retriever"].invoke(question)

        """-----------outputs-----------"""
        return {
            "documents": documents,
            "question": question,
            "steps": steps
        }

    def _grade_documents(self,state):
        """
        Determines whether the retrieved documents are relevant to the question. Store all relevant documents to the documents dictionary. 
        However, if there is even one irrelevant document, then websearch will be invoked.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        print("---Grading Retrieved Documents---")
        """-----------inputs-----------"""
        documents = state["documents"]
        question = state["question"]
        steps = state["steps"]

        """-----------actions-----------"""
        steps.append("grade_document_retrieval")
        relevant_docs = []
        search = "No"

        for d in documents:
            score = self.grader_chain.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            if grade == "yes":
                relevant_docs.append(d)
            else:
                search = "Yes"
                continue
        """-----------outputs-----------"""
        return {
            "documents": relevant_docs,
            "question": question,
            "search": search,
            "steps": steps,
        }

    def _decide_to_generate(self,state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        print("---At decision Edge---")
        """-----------inputs-----------"""
        search = state["search"]

        """-----------actions & outputs-----------"""
        if search == "Yes":
            return "search"
        else:
            return "generate"

    def _web_search(self,state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """
        print("---Searching the Web---")
        """-----------inputs-----------"""
        documents = state.get("documents", [])
        question = state["question"]
        steps = state["steps"]

        """-----------actions-----------"""
        steps.append("web_search")
        web_results = self.params["web_search_tool"].invoke({"query": question})
        documents.extend(
            [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in web_results
            ]
        )
        """-----------outputs-----------"""
        return {
            "documents": documents, 
            "question": question, 
            "steps": steps
        }

    def _generate(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---Generating Response---")
        """-----------inputs-----------"""
        documents = state["documents"]
        question = state["question"]
        steps = state["steps"]

        """-----------actions-----------"""
        steps.append("generating sub-answer")
        generation = self.rag_chain.invoke({"documents": documents, "question": question})
        print("Response to subquestion:", generation)

        """-----------outputs-----------"""
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }
    
    def _getRAGChain(self):
        prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an assistant for question-answering tasks.
        {{Below is some context from different sources followed by a user's question. Please answer the question based on ONLY on the provided context.

        Documents: {documents}}} <|eot_id|><|start_header_id|>user<|end_header_id|>

        {{ Question: {question} }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        Answer:
        """,
        input_variables=["question", "documents"],
        )

        #define rag chain
        llm =  self.params["rag_llm"]
        rag_chain = prompt | llm | StrOutputParser()
        return rag_chain

    def _getGraderChain(self):
        prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {document} \n\n

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the user question: {question} \n

        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
        )

        #define retieval grader chain
        llm = self.params["grader_llm"]
        retrieval_grader = prompt | llm | JsonOutputParser()
        return retrieval_grader

    def _getQueryDecomposeChain(self):
        prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an assistant for question-answering tasks.
        Perform query decomposition. Given a user question, break it down into distinct sub questions that \
        you need to answer in order to answer the original question. Response with \"The question needs no decomposition\" when no decomposition is needed.
        Generate questions that explicitly mention the subject by name, avoiding pronouns like 'these,' 'they,' 'he,' 'she,' 'it,' etc. Each question should clearly state the subject to ensure no ambiguity.

        Example 1:
        Question: Is Hamlet more common on IMDB than Comedy of Errors?
        Decompositions:
        How many listings of Hamlet are there on IMDB?
        How many listing of Comedy of Errors is there on IMDB?

        Example 2:
        Question: What is the Capital city of Japan?
        Decompositions:
        The question needs no decomposition

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {user_query} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Decompositions:"
        """,
        input_variables=["user_query"],
        )

        # define query decomposition chain
        llm = self.params["query_decompose_llm"]
        query_decompose = prompt | llm | StrOutputParser()
        return query_decompose
    
    def _transform_query(self,state: dict) -> dict:
        """
        Transform the user_query to produce a list of simple questions.
        This is the first node invoked in the graph, with input user question and empty steps list
        response = agentic_rag.invoke({"user_query": question3, "steps": []})

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a list of re-phrased question
        """
        """-----------inputs-----------"""
        user_query = state["user_query"]
        steps = state["steps"]
        print("User Query:", user_query)
        print("---Decomposing the QUERY---")

        """-----------actions-----------"""
        steps.append("transform_query")
        # Re-write question
        sub_questions = self.query_decompose_chain.invoke({"user_query": user_query})

        #parse sub questions as a list
        list_of_questions = [question.strip() for question in sub_questions.strip().split('\n')]

        if list_of_questions[0] == 'The question needs no decomposition':
            #no query decomposition required
            #return question field as list
            """-----------outputs-----------"""
            return {
                "sub_questions": [user_query], 
                "steps": steps, 
                "user_query": user_query
            }
        else:
            print("Decomposed into the following queries:", list_of_questions)
            return {
                "sub_questions": list_of_questions, 
                "steps": steps, 
                "user_query": user_query
            }

    def _crag_loop(self,state: dict) -> dict:
        """
        Determines whether to invoke CRAG graph call.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        """-----------inputs-----------"""
        questions = state["sub_questions"] #list of questions
        steps = state["steps"]
        user_query = state["user_query"]

        """-----------actions-----------"""
        sub_answers =[]
        steps.append("entering iterative CRAG for sub questions")

        #loop through list of decomposed questions
        for q in questions:
            print("Handling subquestion:", q)
            #enters beggining of CRAG graph -- retrieve node with the following state (question, step)
            response = self.compiled_graph.invoke({"question": q, "steps": steps})["generation"]
            sub_answers.append(response)

        """-----------outputs-----------"""
        return {
                "sub_answers": sub_answers,
                "sub_questions": questions,
                "user_query": user_query
            }

    def _consolidate(self,state: dict) -> dict:
        """
        Generate consolidated final answer to the original question, given 1. the original question and 2. the sub_questions with corresponding sub_answers

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---Consolidating Response---")
        """-----------inputs-----------"""
        answers = state['sub_answers']
        questions = state['sub_questions']
        user_query = state['user_query']

        """-----------actions-----------"""
        steps = state["steps"]
        steps.append("generating final answer")
        qa_pairs = []

        #create a list of the decomposed questions with their corresponding answers
        #this intermediary information is used as context to answer the original user_query via in-context learning / RAG approach
        for i in range(min(len(questions), len(answers))):
            qa_pairs.append({questions[i]: answers[i].strip()})
        print("multi hop context", qa_pairs)
        final_response = self.rag_chain.invoke({"documents": qa_pairs, "question": user_query})
        print("Final Response to Original Query:", final_response)

        """-----------outputs-----------"""
        return {
            "user_query": user_query,
            "final_response": final_response,
            "steps": steps,
            "intermediate_qa": qa_pairs,
        }