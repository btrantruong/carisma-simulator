"""
An agent receives inventory of messages (agent/user object) from the agent_pool_manager 
and post/repost messages that will be shown to their followers
"""

from mpi4py import MPI


def run_agent(
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
        # Receive package that contains (friend ids, messages) from agent_pool_manager
        # Wait for agent pack to process
        user_pack = comm_world.recv(
            source=rank_index["agent_pool_manager"],
            status=status,
        )

        if user_pack == "sigterm":
            break

        # Unpack the agent + incoming messages
        user, in_messages = user_pack

        # in_messages: inventory
        # Add in_messages to newsfeed
        user.newsfeed = in_messages + user.newsfeed

        # Do some actions
        new_msgs, passive_actions = user.make_actions()

        # Repack the agent (updated feed) and actions (messages he produced)
        agent_pack_reply = (user, new_msgs, passive_actions)

        # Send the packet to data manager (wait to be received)
        comm_world.send(agent_pack_reply, dest=rank_index["data_manager"])

    print(f"Agent handler stop @ rank: {rank}")
