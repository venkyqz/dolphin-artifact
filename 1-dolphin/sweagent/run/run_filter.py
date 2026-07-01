from pydantic import BaseModel

from sweagent.agent.agents import MultiStageAgent, AbstractAgent
from sweagent.sop.issue_resolver import IssueResolvingSOP


class RunFilter(BaseModel):
    agent_name: str | None = None

    def filter_agent_by_name(self, agent: AbstractAgent):
        if agent.name == self.agent_name:
            return agent
        if isinstance(agent, IssueResolvingSOP):
            for stage_agent in agent.agents:
                res = self.filter_agent_by_name(stage_agent)
                if res:
                    return res
        elif isinstance(agent, MultiStageAgent):
            for stage_agent in agent.stage_agents:
                res = self.filter_agent_by_name(stage_agent)
                if res:
                    return res
        return None
