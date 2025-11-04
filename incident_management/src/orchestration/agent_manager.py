from typing import Dict, Any, List
from datetime import datetime
from sqlalchemy.orm import Session
from .models import QueueTask, TaskStatus
from .security import AccessControl
from crewai import Task, Agent, Crew, Process
from langchain.llms import OpenAI

class AgentOrchestrator:
    def __init__(self, db_session: Session, access_control: AccessControl):
        self.db = db_session
        self.access_control = access_control
        self.llm = OpenAI()
        
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized AI agents."""
        agents = {
            'triage': Agent(
                name='Triage Agent',
                goal='Accurately classify and prioritize incidents',
                backstory='Expert at incident classification and triage',
                llm=self.llm,
                tools=['classify_incident', 'update_priority']
            ),
            
            'diagnosis': Agent(
                name='Diagnosis Agent',
                goal='Identify root causes of incidents',
                backstory='Expert at technical diagnosis and troubleshooting',
                llm=self.llm,
                tools=['analyze_logs', 'query_knowledge_base']
            ),
            
            'remediation': Agent(
                name='Remediation Agent',
                goal='Develop and execute remediation plans',
                backstory='Expert at incident remediation and system recovery',
                llm=self.llm,
                tools=['generate_remediation_plan', 'execute_remediation']
            ),
            
            'reporting': Agent(
                name='Reporting Agent',
                goal='Generate comprehensive incident reports',
                backstory='Expert at technical documentation and reporting',
                llm=self.llm,
                tools=['generate_report', 'update_knowledge_base']
            ),
            
            'ticketing': Agent(
                name='Ticketing Agent',
                goal='Manage incident tickets effectively',
                backstory='Expert at ticket management and tracking',
                llm=self.llm,
                tools=['create_ticket', 'update_ticket']
            )
        }
        return agents

    async def process_task(self, task: QueueTask) -> bool:
        """Process a single task using appropriate agent."""
        try:
            # Update task status
            task.status = TaskStatus.PROCESSING
            self.db.commit()
            
            # Get appropriate agent
            agents = self._create_agents()
            agent = agents.get(task.agent_type)
            if not agent:
                raise ValueError(f"Unknown agent type: {task.agent_type}")
            
            # Create crew for the task
            crew = Crew(
                agents=[agent],
                tasks=[
                    Task(
                        description=str(task.task_json),
                        agent=agent
                    )
                ],
                process=Process.sequential
            )
            
            # Execute the task
            result = crew.kickoff()
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.metadata = {'result': result}
            self.db.commit()
            
            return True
            
        except Exception as e:
            # Handle failure
            task.status = TaskStatus.FAILED
            task.metadata = {'error': str(e)}
            self.db.commit()
            return False

    async def process_queue(self) -> List[Dict[str, Any]]:
        """Process all pending tasks in the queue."""
        results = []
        
        # Get all pending tasks, ordered by priority
        pending_tasks = (
            self.db.query(QueueTask)
            .filter(QueueTask.status == TaskStatus.PENDING)
            .order_by(QueueTask.priority.desc())
            .all()
        )
        
        for task in pending_tasks:
            success = await self.process_task(task)
            results.append({
                'task_id': task.id,
                'success': success,
                'metadata': task.metadata
            })
            
        return results

    def add_task(self, task_data: Dict[str, Any], agent_type: str, priority: int = 1) -> QueueTask:
        """Add a new task to the queue."""
        task = QueueTask(
            task_json=task_data,
            status=TaskStatus.PENDING,
            agent_type=agent_type,
            priority=priority,
            created_at=datetime.utcnow()
        )
        
        self.db.add(task)
        self.db.commit()
        
        return task