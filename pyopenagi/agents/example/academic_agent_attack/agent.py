from ...react_agent_attack import ReactAgentAttack

class AcademicAgentAttack(ReactAgentAttack):
    def __init__(self,
                 agent_name,
                 task_input,
                 agent_process_factory,
                 log_mode: str,
                 args=None,
                 attacker_tool=None,
                 vector_db=None,
                 agg=None
        ):
        ReactAgentAttack.__init__(self, agent_name, task_input, agent_process_factory, log_mode, args, attacker_tool, vector_db, agg)
        self.workflow_mode = "manual" 

    def run(self):
        return super().run()
