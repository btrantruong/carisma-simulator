from mpi4py import MPI
import time


def run_policy_filter(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,  # If needed for future logic
    rank_index: dict,
):

    # Status of the processes
    status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    while True:

        # Wait for a batch of (agents, in_messages) to process
        user_packs_batch = comm_world.recv(
            source=rank_index["data_manager"], status=status
        )

        # Check for termination signal
        if user_packs_batch == "sigterm":
            comm_world.send("sigterm", dest=rank_index["agent_pool_manager"])
            break

        processed_batch = user_packs_batch

        # Redirect the processed batch to agent pool manager
        comm_world.send(processed_batch, dest=rank_index["agent_pool_manager"])
