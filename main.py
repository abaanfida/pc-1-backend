# main.py (Multi-User UUID Version with State Machine)

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import re
from groq import Groq
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path


# Load environment variables
load_dotenv()

app = FastAPI()

# Allow requests from your Expo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create user_data directory if it doesn't exist
USER_DATA_DIR = Path("user_data")
USER_DATA_DIR.mkdir(exist_ok=True)

@dataclass
class State:
    name: str
    previous_state: Optional[str]
    condition: Optional[str]
    questions: List[str]
    variables: List[str]
    prompt_actions: List[str]
    prompt_fields: List[str]
    variable_actions: List[str]
    next_state: Optional[str]

class StateManager:
    def __init__(self, csv_path: str, user_id: str):
        self.user_id = user_id
        self.user_dir = USER_DATA_DIR / user_id
        self.user_dir.mkdir(exist_ok=True)
        
        self.states = self.parse_state_machine(csv_path)
        self.user_data = {}
        self.current_state = "q1"
        self.processing_log = []
        
        # Load existing user data if available
        self.load_user_session()

    def get_user_file_path(self, filename: str) -> Path:
        """Get the full path for a user-specific file"""
        return self.user_dir / filename

    def load_user_session(self):
        """Load user's session data if it exists"""
        session_file = self.get_user_file_path("session.json")
        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    self.user_data = session_data.get("user_data", {})
                    self.current_state = session_data.get("current_state", "q1")
                    self.processing_log = session_data.get("processing_log", [])
            except Exception as e:
                print(f"Error loading session for user {self.user_id}: {e}")

    def save_user_session(self):
        """Save user's session data"""
        session_file = self.get_user_file_path("session.json")
        try:
            session_data = {
                "user_data": self.user_data,
                "current_state": self.current_state,
                "processing_log": self.processing_log
            }
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving session for user {self.user_id}: {e}")

    def parse_csv_line(self, line: str) -> List[str]:
        parts = []
        current = []
        in_array = False
        for char in line:
            if char == '[':
                in_array = True
            elif char == ']':
                in_array = False
            if char == ',' and not in_array:
                parts.append(''.join(current))
                current = []
            else:
                current.append(char)
        if current:
            parts.append(''.join(current))
        return parts

    def parse_state_machine(self, csv_path: str) -> Dict[str, List[State]]:
        states = {}
        with open(csv_path, 'r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                row = self.parse_csv_line(line.strip())
                state_name = row[0]
                prev_state = row[1] if row[1] != "null" else None
                condition = row[2] if row[2] != "null" else None
                try:
                    questions = json.loads(row[3])
                    variables = json.loads(row[4])
                    actions = json.loads(row[5])
                    p_fields = json.loads(row[6])
                    variable_actions = json.loads(row[7])
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in row: {row}\n{str(e)}")
                    continue
                next_state = row[8] if row[8] != "null" else None
                
                state = State(
                    name=state_name,
                    previous_state=prev_state,
                    condition=condition,
                    questions=questions,
                    variables=variables,
                    prompt_actions=actions,
                    prompt_fields=p_fields,
                    variable_actions=variable_actions,
                    next_state=next_state
                )
                if state_name not in states:
                    states[state_name] = []
                states[state_name].append(state)
        return states

    def replace_markers(self, text: str) -> str:
        # Load JSON data for @markers
        json_data = {}
        if os.path.exists("prompts_with_json.json"):
            with open("prompts_with_json.json", 'r', encoding='utf-8') as file:
                json_data = json.load(file)

        def replace_match(match):
            var_name = match.group(1)
            value = self.user_data.get(var_name, f"UNKNOWN_{var_name}")
            return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)

        def replace_json_match(match):
            var_name = match.group(1)
            value = self.user_data.get(var_name, json_data.get(var_name, f"UNKNOWN_JSON_{var_name}"))
            return json.dumps(value, ensure_ascii=False)

        text = re.sub(r"@(\w+)", replace_json_match, text)
        text = re.sub(r"\^(\w+)", replace_match, text)
        return text

    def evaluate_condition(self, condition: Optional[str]) -> bool:
        if not condition:
            return True
        
        value = self.user_data.get(condition.lstrip("!"), "").strip().lower()
        if condition.startswith("!"):
            return value in ["no", "false", "0", "", None]
        else:
            return value in ["yes", "true", "1"]

    def clean_response(self, raw_response: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
        cleaned = re.sub(r"(?i)(---json|```json|```)", "", cleaned).strip()
        return cleaned

    def run_variable_actions(self, state: State):
        """Execute variable actions for a state"""
        for action in state.variable_actions:
            if "=" in action:
                var, value = action.split("=", 1)
                self.user_data[var] = None if value == "null" else value
                self.processing_log.append(f"Set {var} = {value}")
        self.save_user_session()

    def run_prompt_actions(self, state: State, api_key: str):
        """Execute prompt actions for a state"""
        if not api_key:
            raise Exception("API key is required for prompt actions")
        
        client = Groq(api_key=api_key)
        
        for i, action in enumerate(state.prompt_actions):
            if i >= len(state.prompt_fields):
                self.processing_log.append(f"Warning: No prompt field for action {i}, skipping")
                continue
            
            field_name = state.prompt_fields[i]
            prompt = self.replace_markers(action)
            
            messages = [
                {"role": "system", "content": "You are a concise assistant that provides accurate and helpful responses."},
                {"role": "user", "content": prompt}
            ]
            print(f"Asking:{prompt}")
            response_text = ""
            try:
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.6,
                    max_completion_tokens=4096,
                    top_p=0.95,
                    stream=True,
                )
                
                for chunk in completion:
                    content = chunk.choices[0].delta.content or ""
                    response_text += content

                cleaned_response = self.clean_response(response_text)
                try:
                    self.user_data[field_name] = json.loads(cleaned_response)
                except json.JSONDecodeError:
                    self.user_data[field_name] = cleaned_response
                
                self.processing_log.append(f"✅ Generated {field_name}")
                self.save_user_session()
                
            except Exception as e:
                error_msg = f"Error generating {field_name}: {str(e)}"
                self.processing_log.append(f"❌ {error_msg}")
                raise Exception(error_msg)

    def get_current_state_info(self):
        """Get information about the current state"""
        state_options = self.states.get(self.current_state, [])
        valid_states = [s for s in state_options if self.evaluate_condition(s.condition)]
        
        if not valid_states:
            return None, True
        
        return valid_states[0], False

    def process_current_state(self, api_key: str = None):
        """Process the current state completely"""
        max_iterations = 50
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            selected_state, completed = self.get_current_state_info()
            if completed:
                self.processing_log.append("Form completed - no valid states found")
                self.save_user_session()
                return {
                    "questions": [],
                    "variables": [],
                    "next": None,
                    "completed": True,
                    "processing_log": self.processing_log.copy(),
                    "requires_api": False
                }
            
            if selected_state.variable_actions:
                self.run_variable_actions(selected_state)
            
            if selected_state.prompt_actions:
                if not api_key:
                    return {
                        "questions": selected_state.questions,
                        "variables": selected_state.variables,
                        "next": selected_state.next_state,
                        "completed": False,
                        "processing_log": self.processing_log.copy(),
                        "requires_api": True,
                        "prompt_actions_pending": True
                    }
                
                try:
                    self.run_prompt_actions(selected_state, api_key)
                except Exception as e:
                    return {
                        "error": str(e),
                        "completed": False,
                        "processing_log": self.processing_log.copy(),
                        "requires_api": False
                    }
            
            if selected_state.questions:
                return {
                    "questions": selected_state.questions,
                    "variables": selected_state.variables,
                    "next": selected_state.next_state,
                    "completed": False,
                    "processing_log": self.processing_log.copy(),
                    "requires_api": False
                }
            
            if selected_state.next_state:
                self.current_state = selected_state.next_state
                self.processing_log.append(f"Advanced to state: {self.current_state}")
                self.save_user_session()
            else:
                self.processing_log.append("Form completed - no next state")
                self.save_user_session()
                return {
                    "questions": [],
                    "variables": [],
                    "next": None,
                    "completed": True,
                    "processing_log": self.processing_log.copy(),
                    "requires_api": False
                }
        
        return {
            "error": f"Maximum iterations ({max_iterations}) reached",
            "completed": True,
            "processing_log": self.processing_log.copy(),
            "requires_api": False
        }

    def submit_answers(self, answers: Dict[str, str], api_key: str = None):
        """Submit answers and advance to next state"""
        self.user_data.update(answers)
        self.processing_log.append(f"Updated user data with: {list(answers.keys())}")
        self.save_user_session()
        
        selected_state, completed = self.get_current_state_info()
        if completed:
            return {"completed": True, "processing_log": self.processing_log.copy()}
        
        if selected_state.prompt_actions and api_key:
            try:
                self.run_prompt_actions(selected_state, api_key)
            except Exception as e:
                return {"error": str(e), "processing_log": self.processing_log.copy()}
        
        if not selected_state.next_state or selected_state.next_state == "null":
            self.processing_log.append("Form completed - reached terminal state")
            self.save_user_session()
            return {"completed": True, "processing_log": self.processing_log.copy()}
        
        self.current_state = selected_state.next_state
        self.processing_log.append(f"Advanced to state: {self.current_state}")
        self.save_user_session()
        
        return {"success": True, "processing_log": self.processing_log.copy()}

    def restart(self):
        """Restart the state machine"""
        self.user_data = {}
        self.current_state = "q1"
        self.processing_log = []
        self.save_user_session()

# Dictionary to store state managers per user
user_sessions: Dict[str, StateManager] = {}

def get_or_create_session(user_id: str) -> StateManager:
    """Get existing session or create new one for user"""
    if user_id not in user_sessions:
        csv_path = "state_machine.csv"
        user_sessions[user_id] = StateManager(csv_path, user_id)
    return user_sessions[user_id]

# --- Enhanced Section Dependency Map ---
SECTION_DEPENDENCIES = {
    "projectName": ["districtName", "location", "scope"],
    "districtName": ["location", "scope"],
    "sector": ["Objectives", "technology", "ICT-Reqs"],
    "sponsAgency": ["opAgency", "exeAgency", "maintAgency", "managementStructure"],
    "opAgency": ["exeAgency", "maintAgency", "managementStructure"],
    "duration": ["startDate", "endDate", "budget", "capitalCostEstimates"],
    "startDate": ["endDate", "duration"],
    "budget": ["capitalCostEstimates", "maintenanceCosts", "financialPlanTable"],
    "technology": ["ICT-Reqs", "Objectives", "benefits"],
    "scope": ["location", "Objectives", "benefits"],
    "location": ["districtName", "Supply and Demand"],
    "Objectives": ["benefits", "financialPlanTable", "technology", "scope"],
    "ICT-Reqs": ["technology", "capitalCostEstimates", "managementStructure"],
    "capitalCostEstimates": ["maintenanceCosts", "financialPlanTable", "budget"],
    "maintenanceCosts": ["financialPlanTable", "capitalCostEstimates"],
    "financialPlanTable": ["capitalCostEstimates", "maintenanceCosts", "benefits"],
    "Supply and Demand": ["benefits", "financialPlanTable", "Objectives"],
    "benefits": ["Objectives", "financialPlanTable", "Supply and Demand"],
    "managementStructure": ["sponsAgency", "opAgency", "exeAgency", "maintAgency"],
    "prepared_by": ["checked_by", "approved_by"],
    "checked_by": ["approved_by", "prepared_by"],
    "additionalProjects": ["scope", "Objectives", "budget"],
    "stakeholders": ["managementStructure", "Objectives"]
}

SECTION_LABELS = {
    "projectName": "Project Name",
    "districtName": "District Name",
    "sector": "Sector",
    "sponsAgency": "Sponsoring Agency",
    "opAgency": "Operating Agency",
    "exeAgency": "Executing Agency",
    "maintAgency": "Maintenance Agency",
    "duration": "Project Duration",
    "startDate": "Start Date",
    "endDate": "End Date",
    "scope": "Project Scope",
    "location": "Project Location",
    "stakeholders": "Key Stakeholders",
    "Objectives": "Project Objectives",
    "ICT-Reqs": "ICT Requirements",
    "Supply and Demand": "Supply and Demand Analysis",
    "capitalCostEstimates": "Capital Cost Estimates",
    "maintenanceCosts": "Maintenance Costs",
    "financialPlanTable": "Financial Plan",
    "benefits": "Expected Benefits",
    "managementStructure": "Management Structure",
    "additionalProjects": "Additional Projects",
    "prepared_by": "Prepared By",
    "checked_by": "Checked By",
    "approved_by": "Approved By",
}

@app.post("/review-section")
async def review_section(request: Request):
    """Review and update a section of PC1 JSON and its related sections using LLM."""
    data = await request.json()
    user_id = data.get("user_id")
    section = data.get("section")
    queries = data.get("queries", [])
    api_key = data.get("api_key")

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not api_key:
        return {"error": "API key is required"}

    state_manager = get_or_create_session(user_id)
    file_name = state_manager.get_user_file_path("PC1_Output.json")
    
    if not file_name.exists():
        return {"error": f"PC1_Output.json not found. Please generate JSON first."}

    with open(file_name, "r", encoding="utf-8") as f:
        pc1_data = json.load(f)

    if section not in pc1_data:
        return {"error": f"Section '{section}' not found in JSON."}

    updated_sections = {}

    def update_section(sec_name, original, extra_note=""):
        queries_text = "\n".join([f"- {q}" for q in queries])
        if extra_note:
            queries_text += f"\n- {extra_note}"

        prompt = f"""
You are reviewing the PC-1 document section: **{sec_name}**.

Current content:
{json.dumps(original, ensure_ascii=False, indent=2)}

User requests these changes:
{queries_text}

Please return ONLY the updated JSON value for this section if the input is JSON otherwise keep it in the style it is in.
(valid JSON, no explanations, no markdown).
"""
        
        print(f"Updating section: {sec_name}")

        client = Groq(api_key=api_key)
        messages = [
            {"role": "system", "content": "You are a precise editor that returns ONLY valid JSON values if the input is in JSON otherwise return in the style it is in."},
            {"role": "user", "content": prompt},
        ]

        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=messages,
            temperature=0.5,
            max_completion_tokens=2048,
            top_p=0.95,
        )

        response_text = completion.choices[0].message.content.strip()
        cleaned_response = state_manager.clean_response(response_text)
        
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return cleaned_response

    pc1_data[section] = update_section(section, pc1_data[section])
    updated_sections[section] = pc1_data[section]

    related_sections = SECTION_DEPENDENCIES.get(section, [])
    for related in related_sections:
        if related in pc1_data:
            section_label = SECTION_LABELS.get(related, related)
            pc1_data[related] = update_section(
                related,
                pc1_data[related],
                extra_note=f"Ensure consistency with the updated '{SECTION_LABELS.get(section, section)}' section."
            )
            updated_sections[related] = pc1_data[related]
            print(f"✅ Updated related section: {section_label}")

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(pc1_data, f, ensure_ascii=False, indent=4)

    updated_sections_with_labels = {}
    for sec, content in updated_sections.items():
        label = SECTION_LABELS.get(sec, sec)
        updated_sections_with_labels[label] = content

    return {
        "message": f"Section '{SECTION_LABELS.get(section, section)}' and {len(related_sections)} related sections updated ✅",
        "updated_sections": updated_sections_with_labels,
        "updated_count": len(updated_sections)
    }

@app.get("/get-questions")
async def get_questions(user_id: str):
    """Return the questions for the current state."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    state_manager = get_or_create_session(user_id)
    result = state_manager.process_current_state()
    return result

@app.get("/download-json")
def download_json(user_id: str):
    """Download user's JSON file."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    state_manager = get_or_create_session(user_id)
    file_path = state_manager.get_user_file_path("PC1_Output.json")
    
    if not file_path.exists():
        return {"error": "JSON not found. Please generate first."}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.post("/submit-answers")
async def submit_answers(request: Request):
    """Receive answers and process the current state."""
    data = await request.json()
    user_id = data.get("user_id")
    answers = data.get("answers", {})
    api_key = data.get("api_key")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    state_manager = get_or_create_session(user_id)
    submit_result = state_manager.submit_answers(answers, api_key)
    
    if "error" in submit_result:
        return submit_result
    
    if submit_result.get("completed"):
        return {
            "message": "Form completed",
            "completed": True,
            "processing_log": submit_result["processing_log"]
        }
    
    result = state_manager.process_current_state(api_key)
    result["processing_log"] = submit_result["processing_log"] + result.get("processing_log", [])
    
    return result

@app.post("/process-with-api")
async def process_with_api(request: Request):
    """Process current state that requires API key."""
    data = await request.json()
    user_id = data.get("user_id")
    api_key = data.get("api_key")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not api_key:
        return {"error": "API key is required", "completed": False}
    
    state_manager = get_or_create_session(user_id)
    result = state_manager.process_current_state(api_key)
    return result

@app.post("/generate-json")
async def generate_json(request: Request):
    """Save the user data as JSON."""
    data = await request.json()
    user_id = data.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    state_manager = get_or_create_session(user_id)
    file_path = state_manager.get_user_file_path("PC1_Output.json")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(state_manager.user_data, f, ensure_ascii=False, indent=4)
    
    return {"message": "JSON file generated ✅", "filename": "PC1_Output.json"}

@app.post("/generate-docx")
async def generate_docx(request: Request):
    """Generate the Word document using sample2."""
    data = await request.json()
    user_id = data.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    state_manager = get_or_create_session(user_id)
    file_path = state_manager.get_user_file_path("PC1_Output.docx")
    
    import sample2
    sample2.create_project_document_from_json(state_manager.user_data, str(file_path))
    
    return {"message": "Word document generated ✅", "filename": "PC1_Output.docx"}

@app.get("/download-docx")
def download_docx(user_id: str):
    """Download user's DOCX file."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    state_manager = get_or_create_session(user_id)
    file_path = state_manager.get_user_file_path("PC1_Output.docx")
    
    if not file_path.exists():
        return {"error": "Document not found. Please generate the document first."}
    
    return FileResponse(
        file_path,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        filename="PC1_Output.docx"
    )

@app.post("/restart")
async def restart(request: Request):
    """Restart the form."""
    data = await request.json()
    user_id = data.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    state_manager = get_or_create_session(user_id)
    state_manager.restart()
    print(f"Form restarted for user {user_id} - reset to state q1")
    return {"message": "Form restarted"}

@app.get("/debug-state")
async def debug_state(user_id: str):
    """Get current state information for debugging."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    
    state_manager = get_or_create_session(user_id)
    return {
        "user_id": user_id,
        "current_state": state_manager.current_state,
        "user_data": state_manager.user_data,
        "processing_log": state_manager.processing_log
    }

@app.get("/section-dependencies")
async def get_section_dependencies():
    """Get the section dependency mapping with labels."""
    dependencies_with_labels = {}
    for section, related_sections in SECTION_DEPENDENCIES.items():
        section_label = SECTION_LABELS.get(section, section)
        related_labels = [SECTION_LABELS.get(sec, sec) for sec in related_sections]
        dependencies_with_labels[section_label] = related_labels
    
    return {
        "dependencies": SECTION_DEPENDENCIES,
        "dependencies_with_labels": dependencies_with_labels,
        "section_labels": SECTION_LABELS
    }