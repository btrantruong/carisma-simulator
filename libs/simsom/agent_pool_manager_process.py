"""
An agent pool manager handle the pool of parallel running agent processes.
Main task is to dispatch User/Agent objects to agent processes.
"""

from mpi4py import MPI
import random as rnd


def run_agent_pool_manager(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    print(f"Handler pool manager start @ rank: {rank}")

    # Ranks of all available agent handler
    agent_handlers_ranks = list(range(rank_index["agent_handler"], size))

    # Status of the processes
    status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    while True:
        print(f"Agent pool manager @ rank {rank} waiting for batch...", flush=True)
        # Get data from policy filter
        users_packs_batch = comm_world.recv(
            source=rank_index["policy_filter"],
            status=status,
        )
        print(f"Agent pool manager @ rank {rank} got the batch...", flush=True)
        #print(f"Received batch at rank {rank}: {type(users_packs_batch)} | {users_packs_batch}", flush=True)
        if users_packs_batch is None or users_packs_batch == "":
            print(f" Warning: Received empty batch at rank {rank}. Skipping...", flush=True)
            continue

        # Check for termination
        if users_packs_batch == "sigterm":
            break

        dispatch_requests = []

        # Filter out suspended or terminated users
        active_user_packs = [user_pack for user_pack in users_packs_batch if not (user_pack[0].is_suspended or user_pack[0].is_terminated)]
        print(f"Active user number: {len(active_user_packs)}.", flush=True)
        if not active_user_packs:
            print(f"Warning: No active users to dispatch @ rank {rank}. Possible system stall.", flush=True)
            continue
        # Dispatch all the agent packs
        while active_user_packs:

            # Pick agent pack from the batch
            user_pack = active_user_packs.pop()

            # Pick handler at random with replacement
            handler_rank = rnd.choice(agent_handlers_ranks)
            # NOTE: The same handler could be issued with more than
            # one agent pack at a time that will be processed when ready.
            print(f"User sent. Left user number: {len(active_user_packs)}", flush=True)
            # Non-blocking dispatch
            req = comm_world.isend(
                user_pack,
                dest=handler_rank,
            )

            dispatch_requests.append(req)

        # Wait for all agent packs dispatched
        MPI.Request.waitall(dispatch_requests)

    # Handlers shutdown
    for i in range(rank_index["agent_handler"], size):
        comm_world.send("sigterm", dest=i)

    print(f"Handler pool manager stop @ rank: {rank}")
