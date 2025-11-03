"""Agent Crew AI for ensemble classification using crew-ai library."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from pydantic import BaseModel, Field

from .hybrid import Classification, ContentDomain

class ClassificationResult(BaseModel):
    """Structure for classification results."""
    domain: str
    category: str
    confidence: float
    subcategories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    reasoning: str
    agent_role: str

class CrewClassifier:
    """Classification system using Crew AI framework."""
    
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-4",
        temperature: float = 0.3
    ):
        self.llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            openai_api_key=openai_api_key
        )
        
        # Create specialized agents
        self.technical_expert = Agent(
            role='Technical Expert',
            goal='Accurately classify technical content and identify technical aspects',
            backstory="""You are an expert in technical domains including software development, 
            infrastructure, security, and data science. Your job is to identify and classify 
            technical content with high precision.""",
            tools=[self._create_classification_tool()],
            llm=self.llm,
            verbose=True
        )
        
        self.business_expert = Agent(
            role='Business Expert',
            goal='Identify business-related content and classify business domains',
            backstory="""You are a business domain expert with deep knowledge of sales, 
            marketing, finance, and operations. You classify business-related content 
            and identify key business aspects.""",
            tools=[self._create_classification_tool()],
            llm=self.llm,
            verbose=True
        )
        
        self.legal_expert = Agent(
            role='Legal Expert',
            goal='Analyze content for legal and compliance aspects',
            backstory="""You are a legal and compliance expert who identifies regulatory, 
            privacy, and compliance-related content. You ensure accurate classification 
            of legal materials.""",
            tools=[self._create_classification_tool()],
            llm=self.llm,
            verbose=True
        )
        
        self.context_analyst = Agent(
            role='Context Analyst',
            goal='Analyze overall context and relationships between different aspects',
            backstory="""You are an expert in understanding context and relationships 
            between technical, business, and legal aspects. You provide holistic analysis 
            of content.""",
            tools=[self._create_classification_tool()],
            llm=self.llm,
            verbose=True
        )
        
        self.consensus_manager = Agent(
            role='Consensus Manager',
            goal='Synthesize different perspectives and reach final classification decision',
            backstory="""You are a senior decision maker who synthesizes input from 
            different experts to make final classification decisions. You weigh different 
            perspectives and ensure accurate final classifications.""",
            tools=[self._create_classification_tool()],
            llm=self.llm,
            verbose=True
        )
    
    def _create_classification_tool(self) -> Tool:
        """Create a tool for classification tasks."""
        return Tool(
            name="classify_content",
            func=lambda x: "Classification analysis complete",
            description="""Analyze and classify content based on domain expertise. 
            Provide detailed reasoning for classification decisions."""
        )
    
    def _create_analysis_task(
        self,
        agent: Agent,
        text: str,
        previous_results: Optional[List[Dict]] = None
    ) -> Task:
        """Create an analysis task for an agent."""
        previous = ""
        if previous_results:
            previous = "\nPrevious classifications:\n" + "\n".join(
                f"- {r['agent_role']}: {r['domain']}/{r['category']} ({r['confidence']:.2f})"
                for r in previous_results
            )
        
        return Task(
            description=f"""Analyze this text and provide classification:{previous}

Text: {text}

Provide your analysis as a JSON object with:
- domain: Main domain (technical/business/legal/general)
- category: Specific category within the domain
- confidence: Confidence score (0-1)
- subcategories: List of relevant subcategories
- tags: List of relevant tags
- reasoning: Detailed explanation of your classification
- agent_role: Your role/expertise area

Focus on your area of expertise while considering the full context.""",
            agent=agent
        )
    
    def _create_consensus_task(
        self,
        text: str,
        results: List[Dict]
    ) -> Task:
        """Create a consensus task."""
        return Task(
            description=f"""Review all expert classifications and determine final classification:

Text: {text}

Expert Classifications:
{json.dumps(results, indent=2)}

Provide final classification as JSON with:
- domain: Final domain decision
- category: Final category decision
- confidence: Overall confidence score
- subcategories: Combined relevant subcategories
- tags: Combined relevant tags
- reasoning: Detailed explanation of final decision
- agent_role: "consensus"

Consider:
1. Agreement/disagreement between experts
2. Confidence levels of each classification
3. Strength of reasoning
4. Overall context""",
            agent=self.consensus_manager
        )
    
    def classify(self, text: str) -> Classification:
        """Run crew-based classification process."""
        # Create initial analysis crew
        analysis_crew = Crew(
            agents=[
                self.technical_expert,
                self.business_expert,
                self.legal_expert,
                self.context_analyst
            ],
            tasks=[
                self._create_analysis_task(self.technical_expert, text),
                self._create_analysis_task(self.business_expert, text),
                self._create_analysis_task(self.legal_expert, text),
                self._create_analysis_task(self.context_analyst, text)
            ],
            process=Process.sequential,
            verbose=True
        )
        
        # Get initial results
        results = []
        for result in analysis_crew.kickoff():
            try:
                parsed = json.loads(result)
                results.append(parsed)
            except Exception as e:
                print(f"Error parsing result: {e}")
                continue
        
        # Create consensus task
        consensus_crew = Crew(
            agents=[self.consensus_manager],
            tasks=[self._create_consensus_task(text, results)],
            process=Process.sequential,
            verbose=True
        )
        
        # Get final consensus
        try:
            consensus_result = json.loads(consensus_crew.kickoff()[0])
            
            return Classification(
                domain=ContentDomain(consensus_result["domain"].lower()),
                category=consensus_result["category"],
                confidence=consensus_result["confidence"],
                subcategories=consensus_result.get("subcategories", []),
                tags=consensus_result.get("tags", []),
                metadata={
                    "reasoning": consensus_result["reasoning"],
                    "method": "crew_ai_consensus",
                    "expert_classifications": results
                }
            )
            
        except Exception as e:
            print(f"Error in consensus: {e}")
            # Fallback to highest confidence classification
            if results:
                best_result = max(results, key=lambda x: x["confidence"])
                return Classification(
                    domain=ContentDomain(best_result["domain"].lower()),
                    category=best_result["category"],
                    confidence=best_result["confidence"],
                    subcategories=best_result.get("subcategories", []),
                    tags=best_result.get("tags", []),
                    metadata={
                        "reasoning": best_result["reasoning"],
                        "method": "crew_ai_fallback",
                        "expert_classifications": results
                    }
                )
            
            return Classification(
                domain=ContentDomain.UNKNOWN,
                category="unknown",
                confidence=0.0,
                metadata={"method": "crew_ai_error"}
            )

# Example usage
if __name__ == "__main__":
    import os
    
    classifier = CrewClassifier(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4"
    )
    
    # Example text
    text = """
    We need to implement OAuth2 authentication for our REST API endpoints
    to comply with SOC2 requirements and protect customer data. This is
    part of our Q4 security initiative and has a budget of $50,000.
    """
    
    result = classifier.classify(text)
    print(f"Final Classification: {result.model_dump_json(indent=2)}")