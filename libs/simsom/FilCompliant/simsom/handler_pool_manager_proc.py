# TODO: Can we change the name to `agent_pool_manager`? 
from mpi4py import MPI
import random as rnd


def run_handler_pool_manager(
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

        # Get data from policy filter
        agent_packs_batch = comm_world.recv(
            source=rank_index["policy_filter"],
            status=status,
        )

        # Check for termination
        if agent_packs_batch == "sigterm":
            break

        dispatch_requests = []

        # Dispatch all the agent packs
        while agent_packs_batch:

            # Pick agent pack from the batch
            agent_pack = agent_packs_batch.pop()

            # Pick handler at random with replacement
            handler_rank = rnd.choice(agent_handlers_ranks)
            # NOTE: The same handler could be issued with more than
            # one agent pack at a time that will be processed when ready.

            # Non-blocking dispatch
            req = comm_world.isend(
                agent_pack,
                dest=handler_rank,
            )

            dispatch_requests.append(req)

        # Wait for all agent packs dispatched
        MPI.Request.waitall(dispatch_requests)

    # Handlers shutdown
    for i in range(rank_index["agent_handler"], size):
        comm_world.send("sigterm", dest=i)

    print(f"Handler pool manager stop @ rank: {rank}")
