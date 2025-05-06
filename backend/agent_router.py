from langchain.agents import initialize_agent, AgentType

def route_to_agent(query, role, tools, backend_tools, llm):
    agent = initialize_agent(
        tools.get(role, []) + backend_tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    return agent.run(query)
