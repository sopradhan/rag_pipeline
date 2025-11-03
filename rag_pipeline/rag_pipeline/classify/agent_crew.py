"""Agent Crew AI for ensemble classification decisions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .hybrid import Classification, ContentDomain
from .llm_classifier import LLMClassifierConfig

class AgentRole(str, Enum):
    """Specialized agent roles."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    LEGAL = "legal"
    CONTEXT = "context"
    CONSENSUS = "consensus"

@dataclass
class ClassificationVote:
    """Vote from an individual agent."""
    agent_role: AgentRole
    classification: Classification
    confidence: float
    reasoning: str
    weight: float = 1.0

class AgentCrewConfig:
    """Configuration for the Agent Crew system."""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        min_confidence: float = 0.6,
        weights: Optional[Dict[AgentRole, float]] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.min_confidence = min_confidence
        
        # Default weights for different agents
        self.weights = weights or {
            AgentRole.TECHNICAL: 1.0,
            AgentRole.BUSINESS: 1.0,
            AgentRole.LEGAL: 1.0,
            AgentRole.CONTEXT: 1.2,  # Context agent gets higher weight
            AgentRole.CONSENSUS: 1.5  # Consensus agent gets highest weight
        }
        
        # Prompts for different agents
        self.prompts = {
            AgentRole.TECHNICAL: """Analyze this text from a technical perspective:
            {text}
            
            Previous classifications:
            {classifications}
            
            Evaluate the technical aspects and provide:
            1. Is this primarily technical content?
            2. What technical categories are relevant?
            3. How confident are you in this assessment?
            
            Respond in JSON format:
            {
                "is_technical": bool,
                "categories": [str],
                "confidence": float,
                "reasoning": str
            }""",
            
            AgentRole.BUSINESS: """Analyze this text from a business perspective:
            {text}
            
            Previous classifications:
            {classifications}
            
            Evaluate the business aspects and provide:
            1. Is this primarily business content?
            2. What business categories are relevant?
            3. How confident are you in this assessment?
            
            Respond in JSON format:
            {
                "is_business": bool,
                "categories": [str],
                "confidence": float,
                "reasoning": str
            }""",
            
            AgentRole.LEGAL: """Analyze this text from a legal/compliance perspective:
            {text}
            
            Previous classifications:
            {classifications}
            
            Evaluate the legal aspects and provide:
            1. Is this primarily legal content?
            2. What legal categories are relevant?
            3. How confident are you in this assessment?
            
            Respond in JSON format:
            {
                "is_legal": bool,
                "categories": [str],
                "confidence": float,
                "reasoning": str
            }""",
            
            AgentRole.CONTEXT: """Analyze the overall context of this text:
            {text}
            
            Previous classifications:
            {classifications}
            
            Consider:
            1. What is the primary purpose/context?
            2. How do different aspects (technical/business/legal) relate?
            3. What is the most relevant classification considering context?
            
            Respond in JSON format:
            {
                "primary_context": str,
                "domain": str,
                "category": str,
                "confidence": float,
                "reasoning": str
            }""",
            
            AgentRole.CONSENSUS: """Review all agent classifications and reach consensus:
            
            Text: {text}
            
            Agent votes:
            {votes}
            
            Previous classifications:
            {classifications}
            
            Determine:
            1. Final domain and category
            2. Confidence level
            3. Key factors for decision
            
            Respond in JSON format:
            {
                "domain": str,
                "category": str,
                "confidence": float,
                "subcategories": [str],
                "tags": [str],
                "reasoning": str
            }"""
        }

class AgentCrew:
    """Ensemble of specialized agents for classification decisions."""
    
    def __init__(
        self,
        config: Optional[AgentCrewConfig] = None,
        api_key: Optional[str] = None
    ):
        self.config = config or AgentCrewConfig()
        
        # Initialize LLM for agents
        self.llm = ChatOpenAI(
            temperature=self.config.temperature,
            model_name=self.config.model_name,
            openai_api_key=api_key
        )
        
        # Create agent chains
        self.agent_chains = {
            role: LLMChain(
                llm=self.llm,
                prompt=PromptTemplate(
                    template=prompt,
                    input_variables=["text", "classifications"]
                )
            )
            for role, prompt in self.config.prompts.items()
            if role != AgentRole.CONSENSUS
        }
        
        # Separate chain for consensus agent
        self.consensus_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=self.config.prompts[AgentRole.CONSENSUS],
                input_variables=["text", "votes", "classifications"]
            )
        )
    
    def _format_classifications(self, classifications: List[Classification]) -> str:
        """Format previous classifications for prompt."""
        formatted = []
        for i, c in enumerate(classifications, 1):
            formatted.append(f"{i}. Domain: {c.domain}, Category: {c.category}")
            formatted.append(f"   Confidence: {c.confidence:.2f}")
            if c.metadata.get("reasoning"):
                formatted.append(f"   Reasoning: {c.metadata['reasoning']}")
            formatted.append("")
        return "\n".join(formatted)
    
    def _format_votes(self, votes: List[ClassificationVote]) -> str:
        """Format agent votes for consensus prompt."""
        formatted = []
        for vote in votes:
            formatted.append(f"Agent {vote.agent_role}:")
            formatted.append(f"- Classification: {vote.classification.domain}/{vote.classification.category}")
            formatted.append(f"- Confidence: {vote.confidence:.2f}")
            formatted.append(f"- Reasoning: {vote.reasoning}")
            formatted.append("")
        return "\n".join(formatted)
    
    def get_agent_votes(
        self,
        text: str,
        previous_classifications: List[Classification]
    ) -> List[ClassificationVote]:
        """Get votes from all specialized agents."""
        votes = []
        formatted_classifications = self._format_classifications(previous_classifications)
        
        for role, chain in self.agent_chains.items():
            try:
                result = chain.run(
                    text=text,
                    classifications=formatted_classifications
                )
                parsed = json.loads(result)
                
                # Create classification based on agent role
                if role == AgentRole.TECHNICAL and parsed.get("is_technical"):
                    domain = ContentDomain.TECHNICAL
                    category = parsed["categories"][0] if parsed["categories"] else "other"
                elif role == AgentRole.BUSINESS and parsed.get("is_business"):
                    domain = ContentDomain.BUSINESS
                    category = parsed["categories"][0] if parsed["categories"] else "other"
                elif role == AgentRole.LEGAL and parsed.get("is_legal"):
                    domain = ContentDomain.LEGAL
                    category = parsed["categories"][0] if parsed["categories"] else "other"
                elif role == AgentRole.CONTEXT:
                    domain = ContentDomain(parsed["domain"].lower())
                    category = parsed["category"]
                else:
                    continue
                
                classification = Classification(
                    domain=domain,
                    category=category,
                    confidence=parsed["confidence"],
                    metadata={"reasoning": parsed["reasoning"]}
                )
                
                votes.append(ClassificationVote(
                    agent_role=role,
                    classification=classification,
                    confidence=parsed["confidence"],
                    reasoning=parsed["reasoning"],
                    weight=self.config.weights[role]
                ))
                
            except Exception as e:
                print(f"Error in {role} agent: {e}")
                continue
        
        return votes
    
    def get_consensus(
        self,
        text: str,
        votes: List[ClassificationVote],
        previous_classifications: List[Classification]
    ) -> Classification:
        """Get final consensus classification."""
        try:
            result = self.consensus_chain.run(
                text=text,
                votes=self._format_votes(votes),
                classifications=self._format_classifications(previous_classifications)
            )
            parsed = json.loads(result)
            
            return Classification(
                domain=ContentDomain(parsed["domain"].lower()),
                category=parsed["category"],
                confidence=parsed["confidence"],
                subcategories=parsed.get("subcategories", []),
                tags=parsed.get("tags", []),
                metadata={
                    "reasoning": parsed["reasoning"],
                    "method": "agent_crew_consensus",
                    "agent_votes": [
                        {
                            "role": v.agent_role,
                            "confidence": v.confidence,
                            "reasoning": v.reasoning
                        }
                        for v in votes
                    ]
                }
            )
            
        except Exception as e:
            print(f"Error in consensus agent: {e}")
            # Fallback to weighted voting
            return self._weighted_voting_consensus(votes)
    
    def _weighted_voting_consensus(
        self,
        votes: List[ClassificationVote]
    ) -> Classification:
        """Fallback consensus using weighted voting."""
        if not votes:
            return Classification(
                domain=ContentDomain.UNKNOWN,
                category="unknown",
                confidence=0.0,
                metadata={"method": "fallback_no_votes"}
            )
        
        # Weight votes by confidence and agent weight
        weighted_votes = {}
        for vote in votes:
            key = (vote.classification.domain, vote.classification.category)
            weight = vote.confidence * vote.weight
            if key in weighted_votes:
                weighted_votes[key] += weight
            else:
                weighted_votes[key] = weight
        
        # Get classification with highest weighted votes
        best_key = max(weighted_votes.items(), key=lambda x: x[1])[0]
        total_weight = sum(weighted_votes.values())
        confidence = weighted_votes[best_key] / total_weight
        
        return Classification(
            domain=best_key[0],
            category=best_key[1],
            confidence=confidence,
            metadata={
                "method": "weighted_voting",
                "agent_votes": [
                    {
                        "role": v.agent_role,
                        "confidence": v.confidence,
                        "reasoning": v.reasoning
                    }
                    for v in votes
                ]
            }
        )
    
    def classify(
        self,
        text: str,
        previous_classifications: Optional[List[Classification]] = None
    ) -> Classification:
        """Get final classification using all agents."""
        previous_classifications = previous_classifications or []
        
        # Get votes from specialized agents
        votes = self.get_agent_votes(text, previous_classifications)
        
        # Get consensus
        consensus = self.get_consensus(text, votes, previous_classifications)
        
        return consensus

# Example usage
if __name__ == "__main__":
    config = AgentCrewConfig(
        model_name="gpt-4",
        temperature=0.3,
        min_confidence=0.6
    )
    
    crew = AgentCrew(config=config)
    
    # Example text
    text = """
    We need to implement OAuth2 authentication for our REST API endpoints
    to comply with SOC2 requirements and protect customer data. This is
    part of our Q4 security initiative and has a budget of $50,000.
    """
    
    result = crew.classify(text)
    print(f"Final Classification: {result.model_dump_json(indent=2)}")