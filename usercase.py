import streamlit as st
import os
import tempfile
import pickle
import hashlib
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
import sys
import json
import re
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import io

# Fix for PyTorch/Streamlit compatibility issue
try:
    import torch
    if hasattr(torch, '_classes') and hasattr(torch._classes, '__path__'):
        if not hasattr(torch._classes.__path__, '_path'):
            torch._classes.__path__._path = []
except ImportError:
    pass

# Import the updated chatbot components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq.chat_models import ChatGroq
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("faiss").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIProductAnalyst:
    """AI Product Analyst Assistant - Specialized for analyzing business requirements and generating structured use cases"""
    
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []
        self.use_cases_registry = {}  # Store generated use cases with metadata
        
        # Enhanced templates following the exact specification
        self.requirement_analysis_template = """
        You are an expert AI Product Analyst. Analyze the following business requirement document and identify all distinct functional requirements that should be converted to separate use cases.

        Your task:
        1. Read and understand the business requirements thoroughly
        2. Identify distinct functional features/modules that need separate use cases
        3. Classify each requirement by priority and complexity
        4. Suggest logical groupings if multiple related requirements exist

        Document Content:
        {document_text}

        Please respond in JSON format:
        {{
            "analysis_summary": "Brief overview of what the document contains",
            "total_features_identified": number,
            "requirements": [
                {{
                    "id": 1,
                    "title": "Concise feature title",
                    "description": "Detailed description of what this feature does",
                    "business_value": "Why this feature is needed",
                    "priority": "High/Medium/Low",
                    "complexity": "Simple/Medium/Complex",
                    "category": "Authentication/Payment/Reporting/etc",
                    "actors_involved": ["Primary actor", "Secondary actor"],
                    "systems_affected": ["System1", "System2"],
                    "dependencies": ["Prerequisite1", "Prerequisite2"]
                }}
            ],
            "clarification_needed": ["Any questions you have about ambiguous requirements"]
        }}
        """
        
        self.use_case_generation_template = """
        You are an expert AI Product Analyst. Convert the following specific business requirement into a detailed use case following this EXACT structure:

        ## Story Title:
        [Generate a concise and descriptive title that reflects the main functionality or goal]

        ## Actor:
        [Identify the primary user or system interacting with the feature]

        ## Story Description:
        As a [actor], I should [do something] in order to [achieve business value].

        ## Pre-requisites:
        [List all dependencies, assumptions, required configurations, or data needed before this use case can be executed]

        ## System:
        [Mention all systems, services, APIs, or components that will participate in or be affected by this use case]

        ## Acceptance Criteria - Steps/Actions:
        [List the detailed, numbered steps the system/user must perform. Include logic, conditions, UI behavior, data changes, etc.]

        ## Acceptance Criteria - Edge Cases and Validations:
        [Include potential edge cases, invalid inputs, and how the system should handle those]

        ## Acceptance Criteria - Tracking and Instrumentation:
        [Specify logs, metrics, events, or monitoring requirements needed to observe this use case in action]

        ## Failure Scenarios:
        [Identify what can go wrong (e.g. service timeout, invalid state, user error) and how it should be gracefully handled]

        Business Requirement Details:
        Title: {requirement_title}
        Description: {requirement_description}
        Business Value: {business_value}
        Priority: {priority}
        Category: {category}
        Actors: {actors}
        Systems: {systems}
        Dependencies: {dependencies}

        Context from previous conversation:
        {conversation_context}

        Generate the use case following the exact structure above. Be detailed and specific in each section.
        """
        
        self.use_case_update_template = """
        You are an expert AI Product Analyst. Based on the user's feedback and our conversation history, update the existing use case while maintaining the same structure.

        Previous Conversation Context:
        {conversation_context}

        Current Use Case:
        {current_use_case}

        User Update Request:
        {update_request}

        Please provide the updated use case following the EXACT same structure:
        ## Story Title:
        ## Actor:
        ## Story Description:
        ## Pre-requisites:
        ## System:
        ## Acceptance Criteria - Steps/Actions:
        ## Acceptance Criteria - Edge Cases and Validations:
        ## Acceptance Criteria - Tracking and Instrumentation:
        ## Failure Scenarios:

        Incorporate the requested changes while maintaining consistency and improving clarity.
        """
        
        self.clarification_template = """
        You are an expert AI Product Analyst. The business requirements provided need clarification before generating accurate use cases.

        Document: {document_text}
        Previous Analysis: {previous_analysis}

        Areas needing clarification:
        {clarification_points}

        Please ask specific, focused questions to gather the missing information needed to create comprehensive use cases. Focus on:
        - Ambiguous functional requirements
        - Missing actor definitions
        - Unclear business rules
        - Integration points that need specification
        - Performance or security requirements
        """

    def add_to_conversation_history(self, user_input: str, analyst_response: str, context: Dict[str, Any] = None):
        """Add interaction to conversation history with metadata"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "analyst_response": analyst_response,
            "context": context or {},
            "interaction_type": context.get("type", "general") if context else "general"
        })
        
        # Keep last 15 interactions to manage context size
        if len(self.conversation_history) > 15:
            self.conversation_history = self.conversation_history[-15:]

    def get_conversation_context(self) -> str:
        """Generate formatted conversation context for prompts"""
        if not self.conversation_history:
            return "No previous conversation context."
        
        context_parts = []
        for i, interaction in enumerate(self.conversation_history[-10:], 1):  # Last 10 interactions
            context_parts.append(f"""
            Interaction {i} ({interaction['timestamp']}):
            User: {interaction['user_input'][:200]}...
            Analyst: {interaction['analyst_response'][:300]}...
            Type: {interaction['interaction_type']}
            """)
        
        return "\n".join(context_parts)

    def analyze_requirements(self, document_text: str) -> Dict[str, Any]:
        """Analyze business requirements document and identify distinct features"""
        try:
            prompt = self.requirement_analysis_template.format(document_text=document_text)
            response = self.llm.invoke(prompt)
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON response
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_text = response_text[json_start:json_end]
                    analysis = json.loads(json_text)
                    
                    # Add to conversation history
                    self.add_to_conversation_history(
                        "Analyze business requirements",
                        f"Identified {analysis.get('total_features_identified', 0)} features for use case generation",
                        {"type": "analysis", "features_count": analysis.get('total_features_identified', 0)}
                    )
                    
                    return analysis
            except json.JSONDecodeError:
                pass
            
            # Fallback parsing if JSON fails
            return self._fallback_requirement_analysis(response_text, document_text)
            
        except Exception as e:
            st.error(f"Error analyzing requirements: {str(e)}")
            return self._create_basic_analysis(document_text)

    def _fallback_requirement_analysis(self, response_text: str, document_text: str) -> Dict[str, Any]:
        """Fallback method when JSON parsing fails"""
        # Extract requirements from structured text
        requirements = []
        lines = response_text.split('\n')
        current_req = {}
        req_id = 1
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['requirement', 'feature', 'functionality', 'story']):
                if current_req and len(current_req.get('title', '')) > 10:
                    current_req['id'] = req_id
                    requirements.append(current_req)
                    req_id += 1
                
                current_req = {
                    'title': line[:100],
                    'description': line,
                    'business_value': 'To be determined',
                    'priority': 'Medium',
                    'complexity': 'Medium',
                    'category': 'General',
                    'actors_involved': ['User'],
                    'systems_affected': ['System'],
                    'dependencies': []
                }
            elif current_req and line:
                current_req['description'] += ' ' + line
        
        if current_req and len(current_req.get('title', '')) > 10:
            current_req['id'] = req_id
            requirements.append(current_req)
        
        return {
            'analysis_summary': 'Requirements analyzed with fallback method',
            'total_features_identified': len(requirements),
            'requirements': requirements,
            'clarification_needed': []
        }

    def _create_basic_analysis(self, document_text: str) -> Dict[str, Any]:
        """Create basic analysis when other methods fail"""
        # Simple text splitting approach
        sections = self._split_document_into_sections(document_text)
        requirements = []
        
        for i, section in enumerate(sections[:10], 1):  # Limit to 10 sections
            if len(section.strip()) > 100:
                requirements.append({
                    'id': i,
                    'title': f'Requirement {i}: {section[:50]}...',
                    'description': section,
                    'business_value': 'Business value to be clarified',
                    'priority': 'Medium',
                    'complexity': 'Medium',
                    'category': 'General',
                    'actors_involved': ['User'],
                    'systems_affected': ['System'],
                    'dependencies': []
                })
        
        return {
            'analysis_summary': 'Basic requirements analysis completed',
            'total_features_identified': len(requirements),
            'requirements': requirements,
            'clarification_needed': ['Please review and provide more specific details for each requirement']
        }

    def _split_document_into_sections(self, text: str) -> List[str]:
        """Split document into logical sections"""
        # Try different splitting strategies
        patterns = [
            r'\n\d+\.\s+',  # Numbered lists
            r'\n[A-Z][^.]*:\s*',  # Titled sections
            r'\n[-*]\s+',  # Bullet points
            r'\n\n+'  # Double newlines
        ]
        
        sections = [text]
        for pattern in patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend([s.strip() for s in parts if s.strip()])
            sections = new_sections
            
            if len(sections) >= 5:  # Stop if we have enough sections
                break
        
        return [s for s in sections if len(s) > 50]  # Filter out very short sections

    def generate_use_case(self, requirement: Dict[str, Any]) -> str:
        """Generate a single use case following the exact specified structure"""
        try:
            conversation_context = self.get_conversation_context()
            
            prompt = self.use_case_generation_template.format(
                requirement_title=requirement.get('title', ''),
                requirement_description=requirement.get('description', ''),
                business_value=requirement.get('business_value', ''),
                priority=requirement.get('priority', 'Medium'),
                category=requirement.get('category', 'General'),
                actors=', '.join(requirement.get('actors_involved', ['User'])),
                systems=', '.join(requirement.get('systems_affected', ['System'])),
                dependencies=', '.join(requirement.get('dependencies', [])),
                conversation_context=conversation_context
            )
            
            response = self.llm.invoke(prompt)
            use_case_content = response.content if hasattr(response, 'content') else str(response)
            
            # Store in registry
            use_case_id = requirement.get('id', len(self.use_cases_registry) + 1)
            self.use_cases_registry[use_case_id] = {
                'id': use_case_id,
                'title': requirement.get('title', f'Use Case {use_case_id}'),
                'content': use_case_content,
                'requirement': requirement,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'version': 1
            }
            
            # Add to conversation history
            self.add_to_conversation_history(
                f"Generate use case for: {requirement.get('title', '')}",
                f"Generated structured use case with all required sections",
                {"type": "generation", "use_case_id": use_case_id}
            )
            
            return use_case_content
            
        except Exception as e:
            error_msg = f"Error generating use case: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def update_use_case(self, use_case_id: int, update_request: str) -> str:
        """Update a specific use case based on user feedback"""
        try:
            if use_case_id not in self.use_cases_registry:
                return f"Use case {use_case_id} not found. Available use cases: {list(self.use_cases_registry.keys())}"
            
            current_use_case_data = self.use_cases_registry[use_case_id]
            conversation_context = self.get_conversation_context()
            
            prompt = self.use_case_update_template.format(
                conversation_context=conversation_context,
                current_use_case=current_use_case_data['content'],
                update_request=update_request
            )
            
            response = self.llm.invoke(prompt)
            updated_content = response.content if hasattr(response, 'content') else str(response)
            
            # Update registry
            self.use_cases_registry[use_case_id].update({
                'content': updated_content,
                'updated_at': datetime.now().isoformat(),
                'version': current_use_case_data['version'] + 1,
                'last_update_request': update_request
            })
            
            # Add to conversation history
            self.add_to_conversation_history(
                f"Update use case {use_case_id}: {update_request}",
                "Use case updated successfully with requested changes",
                {"type": "update", "use_case_id": use_case_id, "version": self.use_cases_registry[use_case_id]['version']}
            )
            
            return updated_content
            
        except Exception as e:
            error_msg = f"Error updating use case {use_case_id}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def regenerate_use_case(self, use_case_id: int, additional_context: str = "") -> str:
        """Regenerate a use case with improvements based on conversation history"""
        try:
            if use_case_id not in self.use_cases_registry:
                return f"Use case {use_case_id} not found."
            
            current_data = self.use_cases_registry[use_case_id]
            requirement = current_data['requirement']
            
            # Enhanced requirement with conversation learnings
            enhanced_requirement = requirement.copy()
            if additional_context:
                enhanced_requirement['description'] += f"\n\nAdditional Context: {additional_context}"
            
            # Generate new version
            new_content = self.generate_use_case(enhanced_requirement)
            
            # Update registry with regeneration info
            self.use_cases_registry[use_case_id].update({
                'content': new_content,
                'updated_at': datetime.now().isoformat(),
                'version': current_data['version'] + 1,
                'regenerated': True,
                'regeneration_context': additional_context
            })
            
            return new_content
            
        except Exception as e:
            error_msg = f"Error regenerating use case {use_case_id}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def ask_clarification(self, document_text: str, analysis: Dict[str, Any]) -> str:
        """Ask for clarification on ambiguous requirements"""
        try:
            clarification_points = analysis.get('clarification_needed', [])
            
            if not clarification_points:
                return "No clarification needed. The requirements are clear enough to proceed with use case generation."
            
            prompt = self.clarification_template.format(
                document_text=document_text[:2000],  # Limit context size
                previous_analysis=json.dumps(analysis, indent=2),
                clarification_points='\n'.join(f"- {point}" for point in clarification_points)
            )
            
            response = self.llm.invoke(prompt)
            clarification_questions = response.content if hasattr(response, 'content') else str(response)
            
            # Add to conversation history
            self.add_to_conversation_history(
                "Request clarification on requirements",
                clarification_questions,
                {"type": "clarification", "points_count": len(clarification_points)}
            )
            
            return clarification_questions
            
        except Exception as e:
            return f"Error generating clarification questions: {str(e)}"

    def generate_all_use_cases(self, document_text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Complete workflow: analyze requirements and generate all use cases"""
        # Step 1: Analyze requirements
        with st.spinner("🔍 Analyzing business requirements..."):
            analysis = self.analyze_requirements(document_text)
        
        if not analysis.get('requirements'):
            st.warning("No requirements identified. Please check your document content.")
            return analysis, []
        
        st.success(f"✅ Analysis complete: {analysis['total_features_identified']} features identified")
        
        # Step 2: Check if clarification is needed
        if analysis.get('clarification_needed'):
            st.info("📋 Some requirements need clarification for better use cases.")
            with st.expander("View Clarification Questions", expanded=True):
                clarification = self.ask_clarification(document_text, analysis)
                st.markdown(clarification)
        
        # Step 3: Generate use cases
        use_cases = []
        requirements = analysis['requirements']
        
        progress_bar = st.progress(0)
        
        for i, requirement in enumerate(requirements):
            with st.spinner(f"🔄 Generating use case {i+1}/{len(requirements)}: {requirement['title'][:50]}..."):
                use_case_content = self.generate_use_case(requirement)
                
                use_cases.append({
                    'id': requirement['id'],
                    'title': requirement['title'],
                    'content': use_case_content,
                    'priority': requirement.get('priority', 'Medium'),
                    'category': requirement.get('category', 'General'),
                    'complexity': requirement.get('complexity', 'Medium'),
                    'actors': requirement.get('actors_involved', ['User']),
                    'systems': requirement.get('systems_affected', ['System']),
                    'business_value': requirement.get('business_value', ''),
                    'requirement': requirement
                })
            
            progress_bar.progress((i + 1) / len(requirements))
        
        progress_bar.empty()
        st.success(f"🎉 Successfully generated {len(use_cases)} use cases!")
        
        return analysis, use_cases

    def get_use_case_by_id(self, use_case_id: int) -> Dict[str, Any]:
        """Retrieve a specific use case by ID"""
        return self.use_cases_registry.get(use_case_id)

    def list_all_use_cases(self) -> List[Dict[str, Any]]:
        """Get all use cases from registry"""
        return list(self.use_cases_registry.values())

    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation for context"""
        if not self.conversation_history:
            return "No conversation history available."
        
        summary_parts = []
        for interaction in self.conversation_history[-5:]:  # Last 5 interactions
            summary_parts.append(f"- {interaction['interaction_type'].title()}: {interaction['user_input'][:50]}...")
        
        return "Recent interactions:\n" + "\n".join(summary_parts)


class EnhancedPDFGenerator:
    """Enhanced PDF generator with professional formatting for use cases"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup professional styles for use case documents"""
        # Title style
        self.title_style = ParagraphStyle(
            'ProductAnalystTitle',
            parent=self.styles['Title'],
            fontSize=28,
            spaceAfter=30,
            textColor=HexColor('#1f4e79'),
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'ProductAnalystSubtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            textColor=HexColor('#4f81bd'),
            alignment=1,
            fontName='Helvetica'
        )
        
        # Use case title style
        self.use_case_title_style = ParagraphStyle(
            'UseCaseTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=15,
            spaceBefore=25,
            textColor=HexColor('#c5504b'),
            borderWidth=2,
            borderColor=HexColor('#c5504b'),
            borderPadding=8,
            fontName='Helvetica-Bold'
        )
        
        # Section heading style
        self.section_heading_style = ParagraphStyle(
            'SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=HexColor('#70ad47'),
            leftIndent=5,
            fontName='Helvetica-Bold'
        )
        
        # Content style
        self.content_style = ParagraphStyle(
            'Content',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=15,
            fontName='Helvetica'
        )
        
        # Metadata style
        self.metadata_style = ParagraphStyle(
            'Metadata',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=5,
            textColor=HexColor('#7f7f7f'),
            fontName='Helvetica'
        )

    def generate_use_cases_pdf(self, analysis: Dict[str, Any], use_cases: List[Dict[str, Any]], 
                              filename: str = "product_analyst_use_cases.pdf") -> bytes:
        """Generate comprehensive PDF with analysis and use cases"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, 
                              topMargin=50, bottomMargin=50)
        
        story = []
        
        # Title page
        story.append(Paragraph("AI Product Analyst", self.title_style))
        story.append(Paragraph("Business Requirements Use Cases", self.subtitle_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Analysis summary
        story.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}", self.metadata_style))
        story.append(Paragraph(f"Total Features Analyzed: {analysis.get('total_features_identified', 0)}", self.metadata_style))
        story.append(Paragraph(f"Use Cases Generated: {len(use_cases)}", self.metadata_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Analysis summary content
        if analysis.get('analysis_summary'):
            story.append(Paragraph("Executive Summary", self.section_heading_style))
            story.append(Paragraph(analysis['analysis_summary'], self.content_style))
            story.append(Spacer(1, 0.2*inch))
        
        story.append(PageBreak())
        
        # Table of contents
        story.append(Paragraph("Table of Contents", self.title_style))
        story.append(Spacer(1, 0.3*inch))
        
        for uc in use_cases:
            priority_symbol = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(uc.get('priority', 'Medium'), "🟡")
            complexity_symbol = {"Simple": "⚪", "Medium": "🟡", "Complex": "🔴"}.get(uc.get('complexity', 'Medium'), "🟡")
            
            story.append(Paragraph(
                f"{priority_symbol} {complexity_symbol} Use Case {uc['id']}: {uc['title']}", 
                self.content_style
            ))
            story.append(Paragraph(
                f"Category: {uc.get('category', 'General')} | Priority: {uc.get('priority', 'Medium')}", 
                self.metadata_style
            ))
            story.append(Spacer(1, 0.1*inch))
        
        story.append(PageBreak())
        
        # Use cases content
        for i, uc in enumerate(use_cases):
            # Use case header with metadata
            priority_symbol = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(uc.get('priority', 'Medium'), "🟡")
            complexity_symbol = {"Simple": "⚪", "Medium": "🟡", "Complex": "🔴"}.get(uc.get('complexity', 'Medium'), "🟡")
            
            story.append(Paragraph(
                f"{priority_symbol} {complexity_symbol} Use Case {uc['id']}: {uc['title']}", 
                self.use_case_title_style
            ))
            
            # Metadata table
            metadata_text = f"""
            <b>Priority:</b> {uc.get('priority', 'Medium')} | 
            <b>Category:</b> {uc.get('category', 'General')} | 
            <b>Complexity:</b> {uc.get('complexity', 'Medium')}<br/>
            <b>Actors:</b> {', '.join(uc.get('actors', ['User']))}<br/>
            <b>Systems:</b> {', '.join(uc.get('systems', ['System']))}
            """
            story.append(Paragraph(metadata_text, self.metadata_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Parse and format use case content
            self._add_use_case_content_to_story(story, uc['content'])
            
            # Add space between use cases
            if i < len(use_cases) - 1:
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def _add_use_case_content_to_story(self, story, content):
        """Parse and add use case content with proper formatting"""
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a section header (starts with ##)
            if line.startswith('## '):
                section_title = line.replace('## ', '').replace(':', '').strip()
                story.append(Paragraph(section_title, self.section_heading_style))
                current_section = section_title
            else:
                # Regular content
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-', '*', '•')):
                    # List item
                    story.append(Paragraph(f"• {line.lstrip('123456789.-* •')}", self.content_style))
                else:
                    # Regular paragraph
                    if line:  # Only add non-empty lines
                        story.append(Paragraph(line, self.content_style))


@st.cache_resource
def get_cached_models(_api_key: str):
    """Initialize models with caching"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 16}
        )
        
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=_api_key,
            temperature=0.1,
            max_tokens=4096
        )
        
        return embedding_model, llm
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None


class ProductAnalystChatbot:
    """Enhanced chatbot specialized for product analysis and use case generation"""
    
    def __init__(self, api_key: str, cache_dir: str = "./analyst_cache"):
        self.api_key = api_key
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize conversation memory
        if 'conversation_memory' not in st.session_state:
            st.session_state.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"
            )
        
        self.memory = st.session_state.conversation_memory
        self._initialize_models()
        
        if self.llm:
            self.analyst = AIProductAnalyst(self.llm)
            self.pdf_generator = EnhancedPDFGenerator()
    
    def _initialize_models(self):
        """Initialize the models"""
        self.embedding_model, self.llm = get_cached_models(self.api_key)

    def _get_file_hash(self, uploaded_files) -> str:
        """Generate hash for uploaded files"""
        hash_content = ""
        for file in uploaded_files:
            hash_content += f"{file.name}_{file.size}"
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def _save_uploaded_files(self, uploaded_files) -> List[str]:
        """Save uploaded files to temporary directory"""
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        return file_paths

    def load_single_pdf_safe(self, pdf_path: str) -> Tuple[List[Any], str]:
        """Load a single PDF file safely"""
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            return docs, None
        except Exception as e:
            return [], str(e)

    def load_pdfs_parallel(self, pdf_files: List[str]) -> List[Any]:
        """Load multiple PDFs in parallel with progress tracking"""
        documents = []
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with ThreadPoolExecutor(max_workers=min(4, len(pdf_files))) as executor:
            future_to_file = {
                executor.submit(self.load_single_pdf_safe, pdf): pdf 
                for pdf in pdf_files
            }
            
            completed = 0
            for future in as_completed(future_to_file):
                pdf_file = future_to_file[future]
                try:
                    docs, error = future.result()
                    if error:
                        errors.append(f"Error loading {os.path.basename(pdf_file)}: {error}")
                    else:
                        documents.extend(docs)
                except Exception as e:
                    errors.append(f"Error loading {os.path.basename(pdf_file)}: {str(e)}")
                
                completed += 1
                progress = completed / len(pdf_files)
                progress_bar.progress(progress)
                status_text.text(f"Processed {completed}/{len(pdf_files)} files...")
        
        progress_bar.empty()
        status_text.empty()
        
        if errors:
            for error in errors:
                st.error(error)
        
        return documents

    def setup_documents(self, uploaded_files) -> bool:
        """Setup documents for analysis"""
        try:
            if not uploaded_files:
                st.error("No files uploaded")
                return False
            
            file_paths = self._save_uploaded_files(uploaded_files)
            
            st.info("📄 Loading PDF files...")
            documents = self.load_pdfs_parallel(file_paths)
            
            if not documents:
                st.error("No documents loaded successfully")
                return False
            
            st.success(f"✅ Loaded {len(documents)} pages from {len(uploaded_files)} files")
            
            # Combine all document text
            all_text = "\n\n".join([doc.page_content for doc in documents])
            st.session_state.document_text = all_text
            st.session_state.documents = documents
            st.session_state.files_processed = True
            
            # Clean up temporary files
            for file_path in file_paths:
                try:
                    os.remove(file_path)
                except:
                    pass
            
            return True
            
        except Exception as e:
            st.error(f"Error setting up documents: {e}")
            return False

    def detect_intent(self, user_input: str) -> Dict[str, Any]:
        """Detect user intent and extract relevant information"""
        user_input_lower = user_input.lower()
        
        # Generate all use cases
        if any(phrase in user_input_lower for phrase in [
            'generate all use cases', 'create all use cases', 'analyze requirements',
            'generate use cases', 'create use cases'
        ]):
            return {"intent": "generate_all", "input": user_input}
        
        # Update specific use case
        update_patterns = [
            r'update\s+use\s*case\s+(\d+)',
            r'modify\s+use\s*case\s+(\d+)', 
            r'change\s+use\s*case\s+(\d+)',
            r'edit\s+use\s*case\s+(\d+)'
        ]
        
        for pattern in update_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                use_case_id = int(match.group(1))
                return {"intent": "update", "use_case_id": use_case_id, "request": user_input}
        
        # Regenerate specific use case
        regen_patterns = [
            r'regenerate\s+use\s*case\s+(\d+)',
            r'remake\s+use\s*case\s+(\d+)',
            r'improve\s+use\s*case\s+(\d+)'
        ]
        
        for pattern in regen_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                use_case_id = int(match.group(1))
                return {"intent": "regenerate", "use_case_id": use_case_id, "context": user_input}
        
        # View specific use case
        view_patterns = [
            r'show\s+use\s*case\s+(\d+)',
            r'display\s+use\s*case\s+(\d+)',
            r'view\s+use\s*case\s+(\d+)',
            r'explain\s+use\s*case\s+(\d+)',
            r'detail\s+use\s*case\s+(\d+)'
        ]
        
        for pattern in view_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                use_case_id = int(match.group(1))
                return {"intent": "view", "use_case_id": use_case_id}
        
        # Clarification response
        if any(phrase in user_input_lower for phrase in [
            'clarification', 'clarify', 'answer is', 'the answer', 'here is the answer'
        ]):
            return {"intent": "clarification", "response": user_input}
        
        # List use cases
        if any(phrase in user_input_lower for phrase in [
            'list use cases', 'show all use cases', 'what use cases',
            'summary', 'overview'
        ]):
            return {"intent": "list", "input": user_input}
        
        # General question
        return {"intent": "question", "input": user_input}

    def process_request(self, user_input: str) -> Tuple[str, Any]:
        """Process user request and return appropriate response"""
        intent_data = self.detect_intent(user_input)
        intent = intent_data["intent"]
        
        try:
            if intent == "generate_all":
                if 'document_text' not in st.session_state:
                    return "Please upload and process business requirement documents first.", None
                
                analysis, use_cases = self.analyst.generate_all_use_cases(st.session_state.document_text)
                st.session_state.current_analysis = analysis
                st.session_state.generated_use_cases = use_cases
                
                response = f"## 🎯 Analysis Complete\n\n"
                response += f"**Features Identified:** {analysis['total_features_identified']}\n"
                response += f"**Use Cases Generated:** {len(use_cases)}\n\n"
                
                if analysis.get('clarification_needed'):
                    response += "### 📋 Clarification Needed\n"
                    response += "Some requirements need clarification for optimal use cases. "
                    response += "Please review the questions below.\n\n"
                
                response += "Use the interface below to view, update, or export your use cases."
                
                return response, {"type": "use_cases", "data": use_cases, "analysis": analysis}
            
            elif intent == "update":
                use_case_id = intent_data["use_case_id"]
                update_request = intent_data["request"]
                
                updated_content = self.analyst.update_use_case(use_case_id, update_request)
                
                # Update session state
                if 'generated_use_cases' in st.session_state:
                    for uc in st.session_state.generated_use_cases:
                        if uc['id'] == use_case_id:
                            uc['content'] = updated_content
                            uc['updated'] = True
                            break
                
                return f"✅ Use Case {use_case_id} has been updated successfully!", {
                    "type": "updated_use_case", 
                    "use_case_id": use_case_id, 
                    "content": updated_content
                }
            
            elif intent == "regenerate":
                use_case_id = intent_data["use_case_id"]
                context = intent_data.get("context", "")
                
                regenerated_content = self.analyst.regenerate_use_case(use_case_id, context)
                
                # Update session state
                if 'generated_use_cases' in st.session_state:
                    for uc in st.session_state.generated_use_cases:
                        if uc['id'] == use_case_id:
                            uc['content'] = regenerated_content
                            uc['regenerated'] = True
                            break
                
                return f"🔄 Use Case {use_case_id} has been regenerated with improvements!", {
                    "type": "regenerated_use_case", 
                    "use_case_id": use_case_id, 
                    "content": regenerated_content
                }
            
            elif intent == "view":
                use_case_id = intent_data["use_case_id"]
                use_case_data = self.analyst.get_use_case_by_id(use_case_id)
                
                if not use_case_data:
                    available_ids = list(self.analyst.use_cases_registry.keys())
                    return f"Use Case {use_case_id} not found. Available use cases: {available_ids}", None
                
                return f"## Use Case {use_case_id} Details", {
                    "type": "single_use_case", 
                    "data": use_case_data
                }
            
            elif intent == "list":
                if not self.analyst.use_cases_registry:
                    return "No use cases have been generated yet. Please generate use cases first.", None
                
                use_cases = self.analyst.list_all_use_cases()
                return "## 📋 All Generated Use Cases", {
                    "type": "use_case_summary", 
                    "data": use_cases
                }
            
            elif intent == "clarification":
                # Handle clarification responses
                return "Thank you for the clarification. You can now ask me to regenerate specific use cases or generate new ones with this additional context.", None
            
            else:
                # General question - provide guidance
                return self._get_guidance_response(user_input), None
                
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(error_msg)
            return error_msg, None

    def _get_guidance_response(self, user_input: str) -> str:
        """Provide guidance based on user input"""
        guidance = """
        ## 🤖 AI Product Analyst Assistant

        I'm here to help you analyze business requirements and generate structured use cases. Here's what I can do:

        ### 📋 **Generate Use Cases**
        - `"Generate all use cases"` - Analyze your documents and create structured use cases
        - `"Analyze requirements"` - Break down your business requirements

        ### 🔍 **View & Navigate**
        - `"Show use case 1"` - View details of a specific use case
        - `"List all use cases"` - See summary of all generated use cases

        ### ✏️ **Modify & Improve**
        - `"Update use case 1 with additional security requirements"` - Modify specific use cases
        - `"Regenerate use case 2"` - Improve a use case with conversation context

        ### 📤 **Export & Download**
        - Use the download buttons to export as PDF, Markdown, or JSON

        **First step:** Upload your business requirement documents and click "Process Files", then ask me to "Generate all use cases"!
        """
        
        return guidance


def display_use_case_summary(use_cases: List[Dict[str, Any]], analysis: Dict[str, Any] = None):
    """Display comprehensive use case summary"""
    st.markdown("### 📊 Use Cases Dashboard")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Use Cases", len(use_cases))
    with col2:
        high_priority = sum(1 for uc in use_cases if uc.get('priority') == 'High')
        st.metric("High Priority", high_priority)
    with col3:
        categories = set(uc.get('category', 'General') for uc in use_cases)
        st.metric("Categories", len(categories))
    with col4:
        updated_count = sum(1 for uc in use_cases if uc.get('updated', False) or uc.get('regenerated', False))
        st.metric("Updated", updated_count)
    
    # Analysis summary
    if analysis:
        with st.expander("📋 Requirements Analysis Summary", expanded=False):
            st.markdown(f"**Analysis:** {analysis.get('analysis_summary', 'N/A')}")
            
            if analysis.get('clarification_needed'):
                st.markdown("**❓ Clarification Needed:**")
                for question in analysis['clarification_needed']:
                    st.markdown(f"• {question}")
    
    # Use cases table
    st.markdown("### 📋 Use Cases Overview")
    
    summary_data = []
    for uc in use_cases:
        priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(uc.get('priority', 'Medium'), "🟡")
        complexity_emoji = {"Simple": "⚪", "Medium": "🟡", "Complex": "🔴"}.get(uc.get('complexity', 'Medium'), "🟡")
        
        status_indicators = []
        if uc.get('updated'):
            status_indicators.append("✏️ Updated")
        if uc.get('regenerated'):
            status_indicators.append("🔄 Regenerated")
        
        status = " ".join(status_indicators) if status_indicators else "📄 Original"
        
        summary_data.append({
            "ID": uc['id'],
            "Priority": f"{priority_emoji} {uc.get('priority', 'Medium')}",
            "Complexity": f"{complexity_emoji} {uc.get('complexity', 'Medium')}",
            "Title": uc['title'],
            "Category": uc.get('category', 'General'),
            "Actors": ", ".join(uc.get('actors', ['User'])[:2]),  # Show first 2 actors
            "Status": status
        })
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Interactive guide
    st.markdown("### 💡 **How to Interact:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **View Details:**
        - `"Show use case 1"` - View full details
        - `"Explain use case 2"` - Detailed explanation
        """)
    
    with col2:
        st.markdown("""
        **Modify & Improve:**
        - `"Update use case 1 with..."` - Add/modify requirements
        - `"Regenerate use case 2"` - Improve with AI
        """)


def display_single_use_case(use_case_data: Dict[str, Any]):
    """Display detailed view of a single use case"""
    uc = use_case_data
    
    # Header with metadata
    priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(uc.get('priority', 'Medium'), "🟡")
    complexity_emoji = {"Simple": "⚪", "Medium": "🟡", "Complex": "🔴"}.get(uc.get('complexity', 'Medium'), "🟡")
    
    st.markdown(f"### {priority_emoji} {complexity_emoji} Use Case {uc['id']}: {uc['title']}")
    
    # Metadata in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Priority:** {uc.get('priority', 'Medium')}")
        st.markdown(f"**Category:** {uc.get('category', 'General')}")
    with col2:
        st.markdown(f"**Complexity:** {uc.get('complexity', 'Medium')}")
        if uc.get('business_value'):
            st.markdown(f"**Business Value:** {uc['business_value'][:50]}...")
    with col3:
        if uc.get('updated'):
            st.success("✏️ Recently Updated")
        if uc.get('regenerated'):
            st.info("🔄 Regenerated")
        
        if uc.get('version', 1) > 1:
            st.markdown(f"**Version:** {uc['version']}")
    
    # Actors and Systems
    if uc.get('actors'):
        st.markdown(f"**👥 Actors:** {', '.join(uc['actors'])}")
    if uc.get('systems'):
        st.markdown(f"**🏗️ Systems:** {', '.join(uc['systems'])}")
    
    st.markdown("---")
    
    # Full use case content
    with st.expander("📄 Complete Use Case Details", expanded=True):
        st.markdown(uc['content'])
    
    # Original requirement context
    if uc.get('requirement'):
        with st.expander("📋 Original Requirement Context", expanded=False):
            req = uc['requirement']
            st.markdown(f"**Description:** {req.get('description', 'N/A')}")
            st.markdown(f"**Business Value:** {req.get('business_value', 'N/A')}")
            if req.get('dependencies'):
                st.markdown(f"**Dependencies:** {', '.join(req['dependencies'])}")
    
    # Action buttons
    st.markdown("### 🔧 Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(f"📋 Copy Use Case {uc['id']}", key=f"copy_{uc['id']}"):
            st.code(uc['content'], language="markdown")
    
    with col2:
        st.download_button(
            label="💾 Download MD",
            data=uc['content'],
            file_name=f"use_case_{uc['id']}_{uc['title'][:20].replace(' ', '_')}.md",
            mime="text/markdown",
            key=f"download_md_{uc['id']}"
        )
    
    with col3:
        json_data = json.dumps(uc, indent=2)
        st.download_button(
            label="📊 Download JSON",
            data=json_data,
            file_name=f"use_case_{uc['id']}.json",
            mime="application/json",
            key=f"download_json_{uc['id']}"
        )
    
    with col4:
        if st.button(f"✏️ Request Update", key=f"update_req_{uc['id']}"):
            st.session_state.update_request_id = uc['id']
    
    # Update request interface
    if st.session_state.get('update_request_id') == uc['id']:
        st.markdown("### ✏️ Update Request")
        update_text = st.text_area(
            "Describe what you want to update or improve:",
            placeholder="e.g., Add data validation requirements, Include error handling for API failures, Add security constraints...",
            key=f"update_text_{uc['id']}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Update", key=f"submit_update_{uc['id']}"):
                if update_text.strip():
                    st.session_state.pending_update = {
                        "use_case_id": uc['id'],
                        "request": f"Update use case {uc['id']}: {update_text}"
                    }
                    st.session_state.update_request_id = None
                    st.rerun()
        
        with col2:
            if st.button("Cancel", key=f"cancel_update_{uc['id']}"):
                st.session_state.update_request_id = None
                st.rerun()


def main():
    st.set_page_config(
        page_title="AI Product Analyst Assistant",
        page_icon="🎯",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f4e79;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #4f81bd;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4f81bd;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">🎯 AI Product Analyst Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Transform Business Requirements into Structured Use Cases</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            help="Get your API key from https://console.groq.com/"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Groq API key to continue")
            st.stop()
        
        st.success("✅ API Key configured!")
        
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload Business Requirement Documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF files containing your business requirements"
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} file(s) ready")
            total_size = sum(file.size for file in uploaded_files) / (1024*1024)
            st.info(f"Total size: {total_size:.1f} MB")
        
        if uploaded_files and st.button("🚀 Process Documents", type="primary"):
            chatbot = ProductAnalystChatbot(api_key)
            if chatbot.setup_documents(uploaded_files):
                st.success("✅ Documents processed successfully!")
                st.session_state.chatbot = chatbot
            else:
                st.error("❌ Failed to process documents")
        
        # Quick actions
        st.header("🎯 Quick Actions")
        if st.session_state.get('files_processed'):
            if st.button("📊 Analyze Requirements", key="analyze_quick"):
                st.session_state.quick_action = "Generate all use cases from the business requirements"
            
            if st.button("📋 List Use Cases", key="list_quick"):
                st.session_state.quick_action = "List all use cases"
        
        # Export options
        if st.session_state.get('generated_use_cases'):
            st.header("📤 Export Options")
            use_cases = st.session_state.generated_use_cases
            analysis = st.session_state.get('current_analysis', {})
            
            try:
                if st.button("📄 Generate PDF Report"):
                    chatbot = st.session_state.get('chatbot')
                    if chatbot:
                        pdf_data = chatbot.pdf_generator.generate_use_cases_pdf(analysis, use_cases)
                        
                        st.download_button(
                            label="💾 Download PDF Report",
                            data=pdf_data,
                            file_name=f"product_analyst_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("📄 PDF report generated!")
            except Exception as e:
                st.error(f"PDF generation error: {str(e)}")
        
        # Help section
        st.header("❓ Help")
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Upload:** Add your business requirement PDFs
            2. **Process:** Click "Process Documents"
            3. **Analyze:** Ask to "Generate all use cases"
            4. **Interact:** View, update, or improve use cases
            5. **Export:** Download as PDF, MD, or JSON
            """)
    
    # Main interface
    if st.session_state.get('files_processed'):
        # Handle pending updates
        if st.session_state.get('pending_update'):
            update_data = st.session_state.pending_update
            del st.session_state.pending_update
            
            if 'chatbot' in st.session_state:
                response, data = st.session_state.chatbot.process_request(update_data['request'])
                
                # Add to chat history
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                
                st.session_state.messages.append({"role": "user", "content": update_data['request']})
                st.session_state.messages.append({"role": "assistant", "content": response, "data": data})
                
                st.rerun()
        
        # Handle quick actions
        if st.session_state.get('quick_action'):
            quick_prompt = st.session_state.quick_action
            del st.session_state.quick_action
            
            if 'chatbot' in st.session_state:
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                
                st.session_state.messages.append({"role": "user", "content": quick_prompt})
                
                response, data = st.session_state.chatbot.process_request(quick_prompt)
                st.session_state.messages.append({"role": "assistant", "content": response, "data": data})
                
                st.rerun()
        
        # Chat interface
        st.header("💬 Interactive Assistant")
        
        # Initialize messages
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display data based on type
                if message.get("data"):
                    data = message["data"]
                    data_type = data.get("type")
                    
                    if data_type == "use_cases":
                        display_use_case_summary(data["data"], data.get("analysis"))
                        
                        # Export options for generated use cases
                        st.markdown("### 💾 Export All Use Cases")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Markdown export
                            all_md = "# Business Requirements Use Cases\n\n"
                            for uc in data["data"]:
                                all_md += f"## Use Case {uc['id']}: {uc['title']}\n\n"
                                all_md += uc['content'] + "\n\n---\n\n"
                            
                            st.download_button(
                                "📝 Download All (MD)",
                                data=all_md,
                                file_name=f"all_use_cases_{len(data['data'])}.md",
                                mime="text/markdown"
                            )
                        
                        with col2:
                            # JSON export
                            json_data = json.dumps(data["data"], indent=2)
                            st.download_button(
                                "📊 Download All (JSON)",
                                data=json_data,
                                file_name=f"all_use_cases_{len(data['data'])}.json",
                                mime="application/json"
                            )
                        
                        with col3:
                            # PDF export handled in sidebar
                            st.info("📄 PDF available in sidebar")
                    
                    elif data_type in ["updated_use_case", "regenerated_use_case"]:
                        uc_id = data["use_case_id"]
                        st.success(f"✅ Use Case {uc_id} has been {'updated' if data_type == 'updated_use_case' else 'regenerated'}!")
                        
                        with st.expander(f"📄 View Updated Use Case {uc_id}", expanded=True):
                            st.markdown(data["content"])
                    
                    elif data_type == "single_use_case":
                        display_single_use_case(data["data"])
                    
                    elif data_type == "use_case_summary":
                        display_use_case_summary(data["data"])
        
        # Chat input
        if prompt := st.chat_input("Ask me to analyze requirements, generate use cases, or modify existing ones..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            if 'chatbot' in st.session_state:
                response, data = st.session_state.chatbot.process_request(prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": response, "data": data})
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
                    # Display data immediately
                    if data:
                        data_type = data.get("type")
                        
                        if data_type == "use_cases":
                            display_use_case_summary(data["data"], data.get("analysis"))
                            
                            # Export options
                            st.markdown("### 💾 Export All Use Cases")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                all_md = "# Business Requirements Use Cases\n\n"
                                for uc in data["data"]:
                                    all_md += f"## Use Case {uc['id']}: {uc['title']}\n\n"
                                    all_md += uc['content'] + "\n\n---\n\n"
                                
                                st.download_button(
                                    "📝 Download All (MD)",
                                    data=all_md,
                                    file_name=f"all_use_cases_{len(data['data'])}.md",
                                    mime="text/markdown",
                                    key="chat_md_export"
                                )
                            
                            with col2:
                                json_data = json.dumps(data["data"], indent=2)
                                st.download_button(
                                    "📊 Download All (JSON)",
                                    data=json_data,
                                    file_name=f"all_use_cases_{len(data['data'])}.json",
                                    mime="application/json",
                                    key="chat_json_export"
                                )
                            
                            with col3:
                                st.info("📄 PDF available in sidebar")
                        
                        elif data_type in ["updated_use_case", "regenerated_use_case"]:
                            uc_id = data["use_case_id"]
                            st.success(f"✅ Use Case {uc_id} has been {'updated' if data_type == 'updated_use_case' else 'regenerated'}!")
                            
                            with st.expander(f"📄 View Updated Use Case {uc_id}", expanded=True):
                                st.markdown(data["content"])
                        
                        elif data_type == "single_use_case":
                            display_single_use_case(data["data"])
                        
                        elif data_type == "use_case_summary":
                            display_use_case_summary(data["data"])
            else:
                st.error("❌ Chatbot not initialized. Please process files first.")
    
    else:
        # Welcome screen with instructions
        st.markdown("## 👋 Welcome to AI Product Analyst Assistant")
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.markdown("""
        **🎯 Purpose:** I specialize in analyzing business requirements and generating well-structured use cases following industry standards.
        
        **📋 What I Do:**
        - Analyze business requirement documents
        - Generate structured use cases with all required sections
        - Support iterative improvements and updates
        - Export professional documentation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Features showcase
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚀 Key Features")
            st.markdown("""
            **📊 Smart Analysis**
            - Identifies distinct functional requirements
            - Classifies by priority and complexity
            - Detects actors and system dependencies
            
            **📝 Structured Generation**
            - Follows exact use case format specification
            - Includes all required sections:
              - Story Title & Description
              - Pre-requisites & Systems
              - Acceptance Criteria (Steps, Edge Cases, Tracking)
              - Failure Scenarios
            
            **🔄 Interactive Improvement**
            - Update specific use cases with natural language
            - Regenerate with conversation context
            - Ask clarification questions for ambiguous requirements
            """)
        
        with col2:
            st.markdown("### 📋 Use Case Structure")
            st.code("""
## Story Title:
[Descriptive title reflecting main functionality]

## Actor:
[Primary user or system]

## Story Description:
As a [actor], I should [action] in order to [business value].

## Pre-requisites:
[Dependencies, assumptions, configurations needed]

## System:
[All systems, services, APIs involved]

## Acceptance Criteria - Steps/Actions:
[Detailed numbered steps with logic and conditions]

## Acceptance Criteria - Edge Cases and Validations:
[Edge cases, invalid inputs, system handling]

## Acceptance Criteria - Tracking and Instrumentation:
[Logs, metrics, events, monitoring requirements]

## Failure Scenarios:
[What can go wrong and graceful handling]
            """, language="markdown")
        
        # Getting started guide
        st.markdown("### 🚀 Getting Started")
        
        tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Setup", "2️⃣ Upload", "3️⃣ Generate", "4️⃣ Interact"])
        
        with tab1:
            st.markdown("""
            ### Initial Setup
            1. **Get Groq API Key**
               - Visit [console.groq.com](https://console.groq.com/)
               - Create account and generate API key
               - Enter key in sidebar
            
            2. **Prepare Documents**
               - Business requirement documents in PDF format
               - Clear, structured content works best
               - Multiple files supported
            """)
        
        with tab2:
            st.markdown("""
            ### Document Upload
            1. **Upload PDFs** using the sidebar file uploader
            2. **Review** uploaded files and total size
            3. **Click "Process Documents"** to analyze content
            4. **Wait** for processing completion
            
            **Supported Content:**
            - Functional requirements
            - Business rules and logic
            - User workflows and processes
            - System integration requirements
            """)
        
        with tab3:
            st.markdown("""
            ### Generate Use Cases
            
            **Primary Command:**
            ```
            "Generate all use cases"
            ```
            
            **What Happens:**
            1. 🔍 **Analysis Phase**
               - Identifies distinct functional requirements
               - Classifies by priority and complexity
               - Detects actors and systems involved
            
            2. 📝 **Generation Phase**
               - Creates structured use cases for each requirement
               - Follows exact format specification
               - Includes comprehensive acceptance criteria
            
            3. ❓ **Clarification Phase** (if needed)
               - Asks specific questions for ambiguous requirements
               - Helps improve use case quality
            """)
        
        with tab4:
            st.markdown("""
            ### Interactive Commands
            
            **View & Navigate:**
            ```
            "Show use case 1"
            "List all use cases"
            "Explain use case 3"
            ```
            
            **Update & Improve:**
            ```
            "Update use case 1 with additional security requirements"
            "Add error handling to use case 2"
            "Regenerate use case 3 with better clarity"
            ```
            
            **Export & Download:**
            - Individual use case downloads (MD, JSON)
            - Bulk export of all use cases
            - Professional PDF reports
            """)
        
        # Example conversation
        st.markdown("### 💬 Example Conversation")
        
        with st.expander("View Complete Workflow Example", expanded=False):
            st.markdown("""
            ```
            👤 User: "Generate all use cases"
            🤖 Bot: 🔍 Analyzing business requirements...
                   ✅ Analysis complete: 4 features identified
                   📊 Dashboard shows 4 use cases with priorities
            
            👤 User: "Show use case 1"
            🤖 Bot: Displays detailed view of Use Case 1 with:
                   - Complete structured content
                   - Metadata (priority, actors, systems)
                   - Action buttons for updates
            
            👤 User: "Update use case 1 with data encryption requirements"
            🤖 Bot: ✅ Use Case 1 has been updated successfully!
                   Shows updated content with encryption details
            
            👤 User: "List all use cases"
            🤖 Bot: 📋 Dashboard with all 4 use cases
                   Shows updated status for Use Case 1
            
            👤 User: Downloads PDF report with all use cases
            ```
            """)
        
        # Requirements and tips
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Requirements")
            st.markdown("""
            **Technical:**
            - Python 3.8+
            - Groq API access
            - Internet connection
            
            **Document Format:**
            - PDF files with text content
            - Business requirements preferred
            - Clear section structure helps
            
            **Best Practices:**
            - Upload comprehensive requirement docs
            - Ask for clarification when needed
            - Iterate and improve use cases
            - Export regularly for backup
            """)
        
        with col2:
            st.markdown("### 💡 Tips for Best Results")
            st.markdown("""
            **Document Preparation:**
            - Include clear functional requirements
            - Specify user types and roles
            - Mention system integrations
            - Define business rules clearly
            
            **Interaction Tips:**
            - Be specific in update requests
            - Use natural language commands
            - Review clarification questions carefully
            - Export and backup your work
            
            **Quality Improvement:**
            - Update use cases based on feedback
            - Regenerate with additional context
            - Review all sections thoroughly
            """)
        
        # Call to action
        st.markdown("### 🚀 Ready to Start?")
        st.info("👈 Upload your business requirement documents in the sidebar and click 'Process Files' to begin!")


if __name__ == "__main__":
    main()