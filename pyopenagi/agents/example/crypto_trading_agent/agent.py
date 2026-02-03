from ...react_agent_attack import ReactAgentAttack

class CryptoTradingAgent(ReactAgentAttack):
    def __init__(self,
                 agent_name,
                 task_input,
                 agent_process_factory,
                 log_mode: str,
                 args,
                 attacker_tool,vector_db, agg
        ):
        ReactAgentAttack.__init__(self, agent_name, task_input, agent_process_factory, log_mode, args, attacker_tool,vector_db, agg)
        self.workflow_mode = args.workflow_mode

    def run(self):
        return super().run()
