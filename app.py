import os
import logging
import json
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Data Models (Unchanged) ---
class ProcedureResult(BaseModel): query_type: str; procedure_name: str; original_message: str
class ProblemDetectionResult(BaseModel): has_multiple_problems: bool; problems: List[Dict[str, str]]; separation_reasoning: str; original_message: str
class ClassificationResult(BaseModel): category: Optional[str]; confidence: float; reasoning: str; need_question: bool; question: Optional[str]

# --- MODIFIED: ConversationState Model ---
class ConversationState(BaseModel):
    current_problems: List[Dict] = []
    current_problem_index: int = 0
    conversation_history: List[Dict] = []
    is_awaiting_followup: bool = False
    is_awaiting_problem_selection: bool = False # NEW state

# --- Prompt Manager (Unchanged, contents folded for brevity) ---
class PromptManager:
    """Centralized management of all system prompts"""
    
    @staticmethod
    def get_procedure_detection_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
            """You are a technical support intent classifier. Determine if the user is:
            A) Describing a technical PROBLEM (needs classification)
            B) Requesting step-by-step GUIDANCE for a procedure
            C) Asking something ambiguous (needs clarification) than treat it as a problem and return problem in query_type

            **PROCEDURAL REQUEST INDICATORS:**
            - "how to", "step by step", "guide me", "show me", "procedure for", "instructions for"
            - Asking for instructions rather than describing symptoms
            - Imperative forms: "tell me...", "explain..."

            **PROBLEM REPORT INDICATORS:**
            - Describes symptoms (crashes, errors, slowness, failures)
            - Uses words like: "problem", "issue", "not working", "broken", "error"
            - Reports unexpected behavior

            **RESPONSE FORMAT:**
            {{
                "query_type": "problem" | "guidance" ,
                "procedure_name": "procedure name if guidance"| "NULL name if problem",
                "original_message": "user input message"
            }}"""), 
            ("human", "User message: {user_message}")
        ])
    
    @staticmethod
    def get_problem_detection_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
            """You are an expert technical support analyzer. Your task is to identify and systematically separate multiple distinct technical problems from user messages.

            **ENHANCED PROBLEM IDENTIFICATION RULES:**
            1. Look for multiple distinct technical issues that affect different systems, applications, or components
            2. Separate problems with different root causes, symptoms, or manifestations
            3. Identify problems occurring in different contexts or timeframes
            4. Consider conjunctions and transitions: "and", "also", "however", "but", "plus", "additionally", "meanwhile", "separately"
            5. Look for enumeration patterns: "first", "second", "another issue", "also have", "on top of that"
            6. Distinguish between primary issues and secondary symptoms of the same root cause

            **SYSTEMATIC PROBLEM SEPARATION EXAMPLES:**
            - "Teams app working slowly and when I start my PC it also starts slow" → 2 DISTINCT problems:
            Problem 1: "Teams application performance is slow during usage"
            Problem 2: "PC boot/startup process is slow"
            
            - "Outlook crashes when I open it and my computer won't boot properly sometimes" → 2 DISTINCT problems:
            Problem 1: "Outlook application crashes upon opening"
            Problem 2: "Computer has intermittent boot/startup failures"

            **SINGLE PROBLEM EXAMPLES (Do NOT separate):**
            - "My computer is running very slow when I open multiple apps" → 1 problem (performance issue)
            - "Teams crashes and shows error message 0x80004005" → 1 problem (specific app crash)
            - "Can't boot my computer, it shows blue screen" → 1 problem (system boot failure)

            **ENHANCED RESPONSE FORMAT:**
            {{
                "has_multiple_problems": true/false,
                "problems": [
                    {{
                        "problem_text": "clear, specific problem description",
                        "context": "additional context and symptoms for this problem",
                        "problem_id": "problem_1/2/3/etc_unique_identifier",
                        "current_user_message": "original user message for this problem"
                    }}
                ],
                "separation_reasoning": "explanation of why problems were separated or kept together",
                "original_message": "full original message"
            }}

            **CRITICAL**: Only separate into multiple problems if they clearly affect different systems/components or have distinctly different root causes. When in doubt, treat as single complex problem."""),
            ("human", "User message: {user_message}")
        ])
    
    @staticmethod
    def get_guidance_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert technical support assistant. The user has requested step-by-step instructions for a procedure. 
            Generate comprehensive, accurate guidance. Use HTML line breaks `<br>` instead of newlines.

            **RESPONSE GUIDELINES:**
            1. Start with a clear title using `<b>` tags.
            2. Provide step-by-step instructions in a clean numbered list format.
            3. Include platform variations (Windows/Mac/Linux/iOS/Android) where applicable.
            4. Add important warnings/cautions where necessary using `<b>Warning:</b>`.
            5. Mention prerequisites if needed.
            6. End with troubleshooting tips or next steps.
            7. Use simple HTML for formatting (e.g., `<br>`, `<b>`). NO MARKDOWN.

            Generate instructions for: {procedure_topic}"""),
            ("human", "User request: {user_message}")
        ])
    
    @staticmethod
    def get_classification_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical support classifier. Your job is to analyze the user's issue and classify it into exactly one of these categories:

            1. hardware
            2. software
            3. performance
            4. app_crash
            5. system_crash

            If the issue clearly fits into one category, set "category" to that value, set "need_question" to false, and set "question" to null. Provide a "confidence" score between 0 and 1 representing how sure you are.
            else if it's unclear which category the issue belongs to, set "category" to null, "confidence" to your best estimate, and "need_question" to true. Then ask ONE concise, relevant follow-up question in the "question" field to clarify the problem context.

            Return your answer strictly in JSON using this schema:
            {{
            "category": string|null,      # one of ["hardware","software","performance","app_crash","system_crash"] or null
            "confidence": number,         # value between 0.0 and 1.0
            "reasoning": "detailed analysis considering conversation context and symptoms",
            "need_question": boolean,     # true if a follow-up question is needed
            "question": string|null       # follow-up question if needed, otherwise null
            }}

            Be concise and only output the JSON object."""),
            ("human", "Problem to classify: {problem_text}\n Conversation: {conversation_history}")
        ])

# --- Core Classifier (Unchanged, contents folded for brevity) ---
class TechnicalSupportClassifier:
    def __init__(self, model: str = "llama3-70b-8192"):
        self.llm = self._initialize_llm()
        self.parser = JsonOutputParser()
        # ... chain initializations ...
        self.procedure_chain=self._build_chain(PromptManager.get_procedure_detection_prompt(),pydantic_model=ProcedureResult)
        self.problem_detection_chain=self._build_chain(PromptManager.get_problem_detection_prompt(),pydantic_model=ProblemDetectionResult)
        self.guidance_chain=self._build_chain(PromptManager.get_guidance_prompt())
        self.classification_chain=self._build_chain(PromptManager.get_classification_prompt(),pydantic_model=ClassificationResult)
    def _initialize_llm(self):
        groq_api_key=os.environ.get("GROQ_API_KEY")
        if not groq_api_key: raise ValueError("Missing GROQ_API_KEY")
        return ChatGroq(model_name="llama3-70b-8192", temperature=0.3, api_key=groq_api_key)
    def _build_chain(self, prompt, pydantic_model=None):
        chain = prompt | self.llm
        if pydantic_model: chain=chain | self.parser | self._create_validator(pydantic_model)
        return chain
    def _create_validator(self, model):
        def validate_output(data):
            try: return model(**data).model_dump()
            except ValidationError as e: raise ValueError(f"Validation error: {e}") from e
        return validate_output
    def detect_request_type(self, user_message): return self.procedure_chain.invoke({"user_message": user_message})
    def detect_problems(self, user_message): return self.problem_detection_chain.invoke({"user_message": user_message})
    def generate_guidance(self, procedure_topic, user_message):
        response=self.guidance_chain.invoke({"procedure_topic": procedure_topic, "user_message": user_message})
        return response.content if hasattr(response, 'content') else str(response)
    def classify_problem(self, problem_text, conversation_history): return self.classification_chain.invoke({"problem_text": problem_text, "conversation_history": json.dumps(conversation_history)})

# --- MODIFIED: ConversationManager ---
class ConversationManager:
    def __init__(self, classifier: TechnicalSupportClassifier):
        self.classifier = classifier
        self.state = ConversationState()

    def reset_state(self):
        self.state = ConversationState()

    def handle_message(self, request_data: Dict) -> Dict[str, Any]:
        """Processes a user message request (now a dict) and returns a structured response."""
        
        # Check for special message types first
        if self.state.is_awaiting_problem_selection and request_data.get("type") == "problem_selection":
            return self._handle_problem_choice(request_data)
        
        user_message = request_data.get("message", "")
        if not user_message:
            return {"type": "error", "content": "No message content received."}

        if self.state.is_awaiting_followup:
            self.state.conversation_history.append({'role': 'user', 'content': user_message})
            self.state.is_awaiting_followup = False
            return self._classify_current_problem()

        # Handle as a new request
        self.reset_state()
        try:
            request_type = self.classifier.detect_request_type(user_message)
            if request_type["query_type"] == "guidance":
                guidance = self.classifier.generate_guidance(request_type["procedure_name"], user_message)
                return {"type": "guidance", "content": guidance}
            else:
                return self._process_problem_report(user_message)
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            return {"type": "error", "content": "Sorry, I encountered an error. Please try again."}

    def _process_problem_report(self, user_message: str) -> Dict[str, Any]:
        """Identifies problems and either classifies or asks user to choose."""
        problem_detection = self.classifier.detect_problems(user_message)
        self.state.current_problems = problem_detection["problems"]
        
        if not self.state.current_problems:
            return {"type": "error", "content": "I couldn't identify a specific problem. Could you rephrase?"}

        if problem_detection["has_multiple_problems"]:
            self.state.is_awaiting_problem_selection = True
            return {
                "type": "multiple_problems_selection",
                "content": "I've found a few issues. Which one would you like to address first?",
                "problems": self.state.current_problems
            }
        else:
            self.state.current_problem_index = 0
            return self._classify_current_problem()

    def _handle_problem_choice(self, request_data: Dict) -> Dict[str, Any]:
        """Handles the user's choice of which problem to tackle first."""
        selected_id = request_data.get("selected_problem_id")
        
        # Reorder self.state.current_problems to put the selected one first
        selected_problem = next((p for p in self.state.current_problems if p['problem_id'] == selected_id), None)
        if selected_problem:
            other_problems = [p for p in self.state.current_problems if p['problem_id'] != selected_id]
            self.state.current_problems = [selected_problem] + other_problems
        
        self.state.is_awaiting_problem_selection = False
        self.state.current_problem_index = 0
        return self._classify_current_problem()

    def _classify_current_problem(self) -> Dict[str, Any]:
        """Classifies the problem at the current index, handling follow-ups and chaining."""
        if self.state.current_problem_index >= len(self.state.current_problems):
            self.reset_state()
            return {"type": "final", "content": "All issues have been addressed! How can I help you further?"}

        problem = self.state.current_problems[self.state.current_problem_index]
        problem_text = problem["current_user_message"]
        
        if not self.state.conversation_history:
            self.state.conversation_history = [{"role": "user", "content": problem_text}]

        try:
            classification = self.classifier.classify_problem(problem_text, self.state.conversation_history)
            
            if classification['need_question']:
                question = classification['question']
                self.state.is_awaiting_followup = True
                self.state.conversation_history.append({'role': 'assistant', 'content': question})
                return {"type": "question", "content": question}
            else:
                category = classification['category']
                response_text = f"For the issue '{problem_text}', I've classified it as: <b>{category.replace('_', ' ').title()}</b>."
                
                # Move to the next problem
                self.state.current_problem_index += 1
                self.state.conversation_history = []
                
                if self.state.current_problem_index < len(self.state.current_problems):
                    next_problem_message = self._classify_current_problem()
                    # Combine the classification result with the next step's message
                    response_text += "<br><br>" + next_problem_message['content']
                    # The type should reflect the final action (e.g., another question or final result)
                    return {"type": next_problem_message['type'], "content": response_text}
                else:
                    self.reset_state()
                    return {"type": "final", "content": response_text + "<br><br>All issues have been addressed. Feel free to ask anything else!"}

        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            return {"type": "error", "content": "I had trouble classifying the problem. Please try rephrasing."}

# --- Flask Application Setup ---
app = Flask(__name__)
try:
    classifier = TechnicalSupportClassifier(model="llama3-70b-8192")
    conversation_manager = ConversationManager(classifier)
except ValueError as e:
    logger.critical(f"Failed to initialize: {e}")
    classifier, conversation_manager = None, None

@app.route("/")
def index(): return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not conversation_manager: return jsonify({"type": "error", "content": "Chatbot not configured."}), 500
    request_data = request.json
    if not request_data: return jsonify({"type": "error", "content": "Invalid request."}), 400
    response = conversation_manager.handle_message(request_data)
    return jsonify(response)

@app.route("/reset", methods=["POST"])
def reset_chat():
    if conversation_manager:
        conversation_manager.reset_state()
        logger.info("Conversation state reset.")
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 500

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"): print("FATAL ERROR: GROQ_API_KEY not set.")
    else: app.run(host='0.0.0.0',debug=True, port=5000)

