from mpi4py import MPI
import random as rnd
import copy
from agent import Agent


def generate_follower_count(max_followers=100):

    alpha = 2.5
    x_min = 1
    r = rnd.random()

    return min(int(x_min * (1 - r) ** (-1 / (alpha - 1))), max_followers)


def batch_message_propagation(message_board, agent, messages):
    batch_size = 1000
    for i in range(0, len(agent.followers), batch_size):
        follower_batch = agent.followers[i : i + batch_size]
        for m in messages:
            for follower_aid in follower_batch:
                message_board[follower_aid].append(copy.deepcopy(m))


def run_data_manager(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    # Example sim params
    num_agents = 1000
    message_count_target = 10000
    b_size = 5

    # Agents init
    agents = [Agent(aid=str(n)) for n in range(num_agents)]

    # Followers
    for a in agents:
        num_followers = generate_follower_count()
        followers = rnd.choices(agents, k=num_followers)
        a.followers = [f.aid for f in followers]
    incoming_messages = {agent.aid: [] for agent in agents}

    # Status of the processes
    status = MPI.Status()

    # DEBUG #
    msgs_store = []

    # Bootstrap sync
    comm_world.Barrier()

    print(f"Data manager start @ rank: {rank}")

    # Batch processing
    message_count = 0
    while message_count < message_count_target:

        batch_send_req = None

        # Unpicked agents count
        n_agents = len(agents)

        if n_agents >= b_size:

            agent_packs_batch = []

            # Build the batch
            for i in range(b_size):

                # Pick an agent at random (without replacement)
                agent_index = rnd.choice(range(n_agents - i))
                picked_agent = agents.pop(agent_index)
                # Get the incoming messages and pack
                agent_pack = (picked_agent, incoming_messages[picked_agent.aid])
                # Add it to the batch
                agent_packs_batch.append(agent_pack)
                # Flush incoming messages
                incoming_messages[picked_agent.aid] = []

            # Non blocking send the batch to the policy_filter
            batch_send_req = comm_world.isend(
                agent_packs_batch,
                dest=rank_index["policy_filter"],
            )

        # Handlers harvesting
        returned_agents = 0
        while returned_agents < b_size:

            # Scan once all the handlers for an agent that completed
            for source in range(rank_index["agent_handler"], size):

                # Check if a handler has done and is waiting
                if comm_world.iprobe(source=source, status=status):

                    # Collect and unpack modified agent and actions
                    mod_agent, new_msgs = comm_world.recv(
                        source=source,
                        status=status,
                    )

                    # Dispatch the messages to agent followers
                    batch_message_propagation(
                        incoming_messages,
                        mod_agent,
                        new_msgs,
                    )

                    # Put the agent back
                    agents.append(mod_agent)
                    returned_agents += 1

                    # Increase counter by the number of action (messages) produced
                    message_count += len(new_msgs)

                    # DEBUG #
                    for m in new_msgs:
                        msgs_store.append(
                            (
                                m.timestamp,
                                f"Agent {mod_agent.aid}, Created: {m.mid} message",
                            )
                        )
                        # NOTE: I'm storing all the messages produced paired with the real creation timestamp

        # Check if batch correctly transmitted
        if batch_send_req:
            batch_send_req.wait()

    # Close policy filter before quitting
    comm_world.send("sigterm", dest=rank_index["policy_filter"])

    print(f"Data manager stop @ rank: {rank}")

    # DEBUG #
    msgs_store.sort(key=lambda x: x[0])  # sort by real timestamp
    for msg in msgs_store:
        print(f"[{msg[0]}] - {msg[1]}")
