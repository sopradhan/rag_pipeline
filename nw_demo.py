"""
Simple demo for `crewai` + `crewai_tools` showing:
- loading .env
- creating a Crew and Agent
- using PDFSearchTool to index/search a small PDF text sample (simulated)
- handling missing LLM clients by falling back to a mock

This demo is safe to run locally and doesn't call external APIs.
"""

from dotenv import load_dotenv
load_dotenv()

try:
    from crewai import Agent, Crew, Process, Task
    from crewai_tools import PDFSearchTool
except Exception as e:
    print('crewai imports not available:', e)
    # create lightweight mocks for demo
    class PDFSearchTool:
        def __init__(self):
            self.docs = {}
        def add_pdf(self, doc_id, text):
            self.docs[doc_id] = text
        def search(self, q):
            # naive substring search
            res = []
            for did, t in self.docs.items():
                if q.lower() in t.lower():
                    res.append({'id': did, 'text': t})
            return res

    class Agent:
        def __init__(self, name='demo'):
            self.name = name
        def run(self, prompt):
            return {'text': f'Mock response for: {prompt}'}

    class Crew:
        def __init__(self):
            self.members = []
        def add(self, a):
            self.members.append(a)

print('Running CrewAI demo...')

# create tools and agent
try:
    pdf_tool = PDFSearchTool()
except Exception as e:
    # Common failure: missing API keys or optional dependencies. Fall back to a safe mock implementation.
    print('PDFSearchTool initialization failed, using mock fallback:', e)
    class PDFSearchTool:
        def __init__(self):
            self.docs = {}
        def add_pdf(self, doc_id, text):
            self.docs[doc_id] = text
        def search(self, q):
            res = []
            for did, t in self.docs.items():
                if q.lower() in t.lower():
                    res.append({'id': did, 'text': t})
            return res
    pdf_tool = PDFSearchTool()

    def create_agent_safe(name='analysis-agent'):
        """Try to instantiate crewai.Agent with common signatures, fall back to a MockAgent."""
        try:
            return Agent(name=name)
        except Exception as e1:
            try:
                return Agent(name=name, role='assistant', goal='Answer queries', backstory='Demo agent')
            except Exception as e2:
                print('Agent creation failed, falling back to MockAgent:', e1, e2)
                class MockAgent:
                    def __init__(self, name='demo'):
                        self.name = name
                    def run(self, prompt):
                        return {'text': f'Mock response for: {prompt}'}
                return MockAgent(name=name)

    agent = create_agent_safe(name='analysis-agent')
    try:
        crew = Crew()
        try:
            crew.add(agent)
        except Exception:
            pass
    except Exception as e:
        print('Crew initialization failed, using MockCrew fallback:', e)
        class MockCrew:
            def __init__(self):
                self.members = []
            def add(self, a):
                self.members.append(a)
        crew = MockCrew()

# Ensure Agent has minimal usable interface; fall back to mock if crewai.Agent requires strict fields
try:
    # simple smoke-check: agent.run should exist
    _ = getattr(agent, 'run')
except Exception:
    class Agent:
        def __init__(self, name='demo'):
            self.name = name
        def run(self, prompt):
            return {'text': f'Mock response for: {prompt}'}
    agent = Agent(name='analysis-agent')

# Simulate adding a PDF (we'll just use text)
pdf_tool.add_pdf('sample_report', 'This report describes an Overheat fault and steps: Check coolant, reduce load, inspect bearings.')

# Query the PDF
query = 'Overheat'
print('Searching PDF for:', query)
results = pdf_tool.search(query)
print('Search results:', results)

# Agent reasoning
prompt = f"Analyze findings: {results[0]['text'] if results else 'no findings'}"
resp = agent.run(prompt)
print('Agent response:', resp)

print('Demo finished.')
