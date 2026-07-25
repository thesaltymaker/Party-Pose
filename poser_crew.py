import requests
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileReadTool, FileWriterTool
from crewai.tools import BaseTool
import os
os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "nomic-embed-text:v1.5"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11435"
os.environ['OPENAI_API_KEY'] = 'sk-dummy-key-not-actually-used'

# --- 1. YOUR SEARCH TOOL IMPLEMENTATION ---
class SearXNGSearchInput(BaseModel):
    query: str

class SearXNGSearchTool(BaseTool):
    name: str = "search"
    description: str = "Search the web for technical documentation and code examples."
    args_schema: type[BaseModel] = SearXNGSearchInput

    def _run(self, query: str) -> str:
        try:
            response = requests.get(
                "http://192.168.2.59:8082/search",
                params={"q": query, "format": "json", "categories": "general,it"},
                timeout=10
            )
            data = response.json()
            results = data.get("results", [])
            if not results: return f"No results for: {query}"
            
            formatted = [f"Results for '{query}':\n"]
            for i, res in enumerate(results[:3], 1): # Top 3 for token efficiency
                formatted.append(f"{i}. {res.get('title')}\n URL: {res.get('url')}\n {res.get('content')[:200]}...")
            return "\n".join(formatted)
        except Exception as e:
            return f"Search error: {str(e)}"

class QuietFileWriter(BaseTool):
    name: str = "file_writer_tool"
    description: str = "Write content to a file. Args: filename (str), content (str), directory (str, optional)"
    
    def _run(self, **kwargs) -> str:
        writer = FileWriterTool()
        writer._run(**kwargs)
        return "File saved successfully."

# Instantiate the tools
searx_tool = SearXNGSearchTool()
file_read = FileReadTool(file_path='poser_spec.md')
file_write = FileWriterTool()
file_writer_tool = FileWriterTool()
#file_writer_tool = QuietFileWriter()
general_file_read_tool = FileReadTool()

# --- 2. AGENT DEFINITIONS ---
#manager_llm = LLM(model="ollama/granite3.3:8b", base_url="http://192.168.2.59:11435", stop=["\nObservation:", "Observation:", "###"  "\nFinal Answer:"], num_ctx=12288, temperature=0)
#manager_llm = LLM(model="ollama/granite3.3:8b", base_url="http://192.168.2.59:11435", num_ctx=20000, temperature=0.1)
architect_llm = LLM(model="ollama/mistral-small3.2:latest",  num_ctx=8192, num_predict=8192, max_tokens=8192, temperature=0.2, base_url="http://192.168.2.59:11434")
heavy_llm = LLM(model="ollama/devstral-small-2:latest",  num_ctx=8192, num_predict=8192, max_tokens=8192, temperature=0.3, base_url="http://192.168.2.59:11434")
python_dev_llm = LLM(model="ollama/devstral-small-2:latest",  num_ctx=8192, num_predict=8192, max_tokens=8192, temperature=0.3, base_url="http://192.168.2.59:11434")
manager_llm = LLM(model="ollama/mistral-small3.2:latest",  num_ctx=16384, num_predict=8192, max_tokens=8192, temperature=0.15, base_url="http://192.168.2.59:11434")

manager = Agent(
    role='Project Manager',
    goal='Execute the Dispatcher Protocol to convert project tasks into coworker assignments.',
    backstory="""You are the Crew Dispatcher. Your sole purpose is to run the following loop:
    
    1. IDENTIFY COWORKER: Using the task description, Choose from [ 'Software Architect', 'Senior Python Developer', 'CUDA Specialist', 'QA Engineer'].
    - If a task involves 'architecture' or 'design', use the 'Software Architect'.
    - If a task involves 'implement' or 'write python', use the 'Senior Python Developer'.
    - If a task involves 'CUDA', 'optimization', or 'bottlenecks', use the 'CUDA Specialist'.
    - If a task involves 'testing', 'debugging', or 'runtime errors', use the 'QA Engineer'.
    2. FORMAT and TOOL CALL:  Call the tool with EXACTLY this format:
        - The "description" field MUST be the COMPLETE original task description. Copy it EXACTLY. Do NOT summarize, shorten, paraphrase, or reword it.
        - The "context" field MUST be the COMPLETE original expected output. Copy it EXACTLY.
        Action: delegate_work_to_coworker
        Action Input: {
            "coworker": "...",
            "task": "...",
            "context": "..."
        }
    3. WAIT for a tool call response.  The tool call response is the completion of the task.  
    4. READ THE RESPONSE. When you receive a response that that contains 'TASK COMPLETED', STOP.  that task is done.
    5. When a coworker returns a result, you must immediately provide a 'Final Answer' stating that the task is complete and summarize the result. Do not re-delegate the same task."
    
    MANAGEMENT RULES:
    1. ALWAYS use the 'delegate_work_to_coworker' tool to get things done.
    2. NEVER Summarize a task description
    3. NEVER answer a technical question yourself.
    4. NEVER Write Code yourself.
    5. NEVER repeat delegation.
    6. YOUR MISSION: Match the task keywords to the specialist's role.
    7. If coworker says "TASK COMPLETED", acknowledge and move to next task
    8. One delegation per task

    Your success is measured by your ability to trigger tool calls instead of providing text answers.
    """,
    allow_delegation=True,
    max_iter=5,
    llm=manager_llm,
    verbose=True
)


architect = Agent(
    role='Software Architect',
    goal='Translate high-level requirements into deep technical design blueprints.',
    backstory="""You are an expert in system design, You understand how to document data flows,
    model pipelines, and resource management in Markdown format. You know how to search the web for best practices. You use a module approach
    You excel at writing the 'architecture.md' file. 

    IMPORTANT: You must ALWAYS use the file_writer_tool to save your work.
    Do NOT include the file contents in your response. The file is already saved.""",
    llm=architect_llm,
    max_iter=3,
    tools=[file_writer_tool, file_read, searx_tool],
    allow_delegation=False
    )

python_dev = Agent(
    role='Senior Python Developer',
    goal='''Implement solution in python. 
        - Write core application logic. 
        - Use a modular approach. 
        - Use search for latest library syntax. 
        - Use high performance patterns to optimize python code performance. 
        - Use multithreading when it optimizes throughput.   
        - Save your work by using the 'file_writer_tool'.  ''',
    backstory='Clean code specialist. You always verify the latest docs via search.',
    llm=heavy_llm,
    tools=[searx_tool, file_write, file_writer_tool, general_file_read_tool],
    allow_delegation=False
)

cuda_specialist = Agent(
    role='CUDA Specialist',
    goal='''Optimize code using CUDA.' \
        - Search for latest GPU kernel best practices.  
        - Search for strategies to avoid CPU - GPU memory transfers. 
        - Read the existing code.  Identify the models to be executed.  
        - Write code for missing execution model implementations
        - Review model inputs and outputs for alignment to model execution. 
        - Prevent unnessecary transfers of memory betweenthe CPU and GPU, prefer code run on the GPU. 
        - You write the code updates.  there are no tools to assiste with optimization like 'optimize_pose_detection' or 'cuda_optimization_analysis'
        - Ensure model executon is CUDA enabled. 
        - Prevent unnessecary transfers of memory betweenthe CPU and GPU, prefer code run on the GPU. 
        - Save your work, python code files by using the 'file_writer_tool'.
        ''',
    backstory='''Performance guru. You use search to find optimized tensor shapes and ops.''',
    llm=heavy_llm,
    tools=[searx_tool, file_write, file_writer_tool, general_file_read_tool],
    allow_delegation=False
)

qa_engineer = Agent(
    role='QA Engineer',
    goal='Test code and search for troubleshooting common errors.',
    backstory='You catch bugs by searching for known edge cases in the libraries used.',
    llm=heavy_llm,
    tools=[searx_tool, file_read, file_writer_tool, general_file_read_tool],
    allow_delegation=False
)

# --- 3. TASK DEFINITIONS ---
t1 = Task(
    description="""Analyze the provided project specification, technical references and if existing, architecture.md to synthesize a high-level modular system architecture. You research and write the architecture design.  There are no external design tools like 'architecture_analysis'
        1. LOGICAL ABSTRACTION: Identify core functional requirements, input constraints, and success metrics from the source documentation. 
        2. HEURISTIC DESIGN: Based on the technical constraints of the chosen frameworks (e.g., MediaPipe), define specific data-driven triggers for system events (such as 'Out of Bounds' or 'Visibility' states). You must establish these thresholds based on best practices for reliability and noise reduction.
        3. SPECIFIC RESOURCES: Identify and recode resouces explicitly named in the specification. 
        4. TRANSFORMATION: Map the raw data flow from sensor input to UI/State output. Include descriptions of coordinate normalization and mapping strategies appropriate for the target display resolution.
        5. FORMAT DESIGN:  place the architecture and desing into the following topics, no additional sections. 
            a. System Architecture
            b. Core Components 
            c. Data Flow
            d. Technical Implementation Details
            e. Component interfaces
            f. Define project file structure  
        6. PERSISTENCE: Execute the 'file_writer_tool' to save your unique design to 'architecture.md'. 
        7. TERMINATION: After the file is saved, respond with ONLY this text:
           Do NOT include the file contents in your response.
           Return "TASK COMPLETED: architecture.md saved."
        """,

    expected_output="""TASK t1 COMPLETED: architecture.md has been saved.""",
        max_iter=2
)
'''A unique, professionally structured 'architecture.md' file that includes:
        - A high-level architectural overview and data-flow diagram description.
        - A logic specification section defining event triggers and sensor thresholds derived from the source specs.
        - A technical implementation table (Input formats, Normalization logic, and Coordinate mapping).
        - DO NOT return the entire file contexts
        - DO return '''
t2 = Task(
    description="""Read the provided project specification.  Read the system architecture.md. 
        - Identify missing components and dependencies.
        - Use a modular approach to Implement and Write python code. 
        - Use the interface design in the architecture.md. 
        - Use search to find current best practices for the libraries needed aligned to the project speciication and architecture.
        - AFTER writing a file, verify it exists and move to the next.""",
    expected_output="Python code files saved to the local directory. Core components are implemented",
    #agent=python_dev,
    context=[t1]
)

t3 = Task(
    description="""Read the existing python code.  Read the architecture.md
        - Review python implementation
        - add or update CUDA optimizations. 
        - write code to assist with determining GPU-specific bottlenecks.""",
    expected_output="Updated Python code files saved to the local directory.  Optimized Python/CUDA code included.",
    #agent=cuda_specialist,
    context=[t2]
)

t4 = Task(
    description="""Final QA. 
        - Compare functions in code and structure to architecture.md to identify deficiencies.
        - Compare application features to project specification to identify deficiencies. 
        - Search for solutions to any runtime errors discovered.
        - write QA report to a file.""",
    expected_output="A bug-free, high-performance application.",
    #agent=qa_engineer,
    context=[t3]
)

# --- 4. THE HIERARCHICAL CREW ---
crew = Crew(
    agents=[architect, python_dev, cuda_specialist, qa_engineer],
    tasks=[ t1, t2, t3, t4],
    process=Process.hierarchical,
    manager_agent=manager,
    memory=True,
    embedder={
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:v1.5",
            "url": "http://localhost:11435",
            "task_type": "search_document",
            "num_ctx": 16384
        }
    },
    verbose=True,
    output_log_file="crew_execution.log" 
)

crew.kickoff()
