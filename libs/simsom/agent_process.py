"""
An agent receives inventory of messages (agent/user object) from the agent_pool_manager 
and post/repost messages that will be shown to their followers
"""

import numpy as np
from mpi4py import MPI


def run_agent(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,
    rank_index: dict,
):

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
        user, in_messages = user_pack  # in_messages: inventory

        # Keep track of the weight of the messages (if a message should appear more than one, it has more weight)
        weight_dict = {}

        # Sort messages and drop duplicates (reshare)
        raw_messages = in_messages + user.newsfeed
        sorted_messages = sorted(raw_messages, key=lambda x: x.time, reverse=True)
        message_filter_dict = {}
        nan_parents = []
        # Iterate to check if there are duplicated reshare messages
        for message in sorted_messages:
            if message.reshared_original_id == np.nan:
                nan_parents.append(message)
            else:
                # check for duplicates and if they are present keep track of the weight (n of time they appear)
                if message.reshared_original_id not in message_filter_dict:
                    message_filter_dict[message.reshared_original_id] = message
                    weight_dict[message.reshared_original_id] = 1
                else:
                    weight_dict[message.reshared_original_id] += 1
        new_newsfeed = list(message_filter_dict.values()) + nan_parents

        # Sort list temporally and based on the weight
        new_newsfeed = sorted(new_newsfeed, key=lambda x: x.time, reverse=True)
        new_newsfeed = sorted(
            new_newsfeed,
            key=lambda x: (weight_dict.get(x.reshared_original_id, 0), x.time),
            reverse=True,
        )

        # replace old newsfeed with filtered newsfeed
        user.newsfeed = new_newsfeed

        # Do some actions
        new_msgs, passive_actions = user.make_actions()

        # Repack the agent (updated feed) and actions (messages he produced)
        agent_pack_reply = (user, new_msgs, passive_actions)

        # Send the packet to data manager (wait to be received)
        comm_world.send(agent_pack_reply, dest=rank_index["data_manager"])
