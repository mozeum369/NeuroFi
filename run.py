from pathlib import Path
from ai_core.agent import SelfModifyingAgent

def main():
    root = Path(__file__).parent.resolve()
    agent = SelfModifyingAgent(project_root=root, name="NeuroFiSelfMod")
    agent.bootstrap()

    # Run a few evolution steps per invocation
    steps = 3
    for _ in range(steps):
        result = agent.evolve_once()
        print(
            f"[evolve] gen={result['generation']}, "
            f"best_skill={result['best_skill']}, "
            f"best_score={result['best_score']:.6f}"
        )

    # Demonstrate controlled self-edit
    agent.self_edit_heartbeat(note="Evolution step complete.")

    # Optional: commit changes if git is available
    agent.git_autocommit(message="Auto-evolve step")

if __name__ == "__main__":
    main()
