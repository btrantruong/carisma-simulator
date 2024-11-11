from mpi4py import MPI


def run_agent_handler(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    print(f"Agent handler start @ rank: {rank}")

    # Status of the processes
    status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    while True:

        # Wait for agent pack to process
        agent_pack = comm_world.recv(
            source=rank_index["agent_pool_manager"],
            status=status,
        )

        if agent_pack == "sigterm":
            break

        # Unpack the agent + incoming messages
        agent, in_messages = agent_pack

        # Add in_messages to newsfeed
        agent.newsfeed = in_messages + agent.newsfeed

        # Do some actions
        new_msgs = agent.make_actions()

        # Repack the agent (updated feed) and actions (messages he produced)
        agent_pack_reply = (agent, new_msgs)

        # Send the packet to data manager (wait to be received)
        comm_world.send(agent_pack_reply, dest=rank_index["data_manager"])

    print(f"Agent handler stop @ rank: {rank}")
