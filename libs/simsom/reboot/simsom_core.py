"""
This module defines a basic schema for the parallel implementation of SimSoM originally
developed by the very sweet Bao Tran Truong from Indiana University.
The definitions of the Agent and Message classes are kept to a minimum
to facilitate code testing. Also, for the same reason, a simple logger is included that keeps track of
simulator activity by generating a file on disk at each execution.

WARNING: Many features are still missing, e.g., messages produced are not saved to disk,
there is no convergence checking, etc. The current code focuses on implementing the agent
pool manager and the mechanism for scheduling agents to each agent handler process when available.

Example of starting command: mpiexec -n 4 python .\simsom_core.py

For any questions except the meaning of life, please contact me at enrico.verdolotti@gmail.com

In hoc signo vinces.
"""

from mpi4py import MPI
import random as rnd
import copy
import sys
from collections import defaultdict
import logging
from datetime import datetime
from tqdm import tqdm
import time

# Configuration constants
RANK_INDEX = {
    "agent_pool_manager": 0,
    "agent_handler": 1,
}


# Configure logging
class SimulationLogger:
    """Custom logger for the social network simulation"""

    def __init__(self, verbose=True, log_file=None):
        self.verbose = verbose

        # Create logger
        self.logger = logging.getLogger("SocialNetSim")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        simple_formatter = logging.Formatter("%(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(
                f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
            file_handler.setFormatter(detailed_formatter)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

        self.progress_bar = None

    def start_progress(self, total):
        """Initialize progress bar"""
        if self.verbose:
            self.progress_bar = tqdm(total=total, desc="Simulation Progress")

    def update_progress(self):
        """Update progress bar"""
        if self.verbose and self.progress_bar:
            self.progress_bar.update(1)

    def close_progress(self):
        """Close progress bar"""
        if self.verbose and self.progress_bar:
            self.progress_bar.close()

    def debug(self, msg):
        """Log debug message"""
        self.logger.debug(msg)

    def info(self, msg):
        """Log info message"""
        self.logger.info(msg)

    def warning(self, msg):
        """Log warning message"""
        self.logger.warning(msg)

    def error(self, msg):
        """Log error message"""
        self.logger.error(msg)


class Message:
    """Represents a message in the social network"""

    def __init__(self, mid, content=None):
        self.mid = mid
        self.content = content
        self.timestamp = time.time()


class Agent:
    """Represents a user/agent in the social network"""

    def __init__(self, aid):
        self.aid = aid
        self.followers = []
        self.newsfeed = []
        self.post_counter = 0
        self.repost_counter = 0

    def make_actions(self, logger=None, cut_off=15):
        """Simulate agent behavior with logging"""

        actions = []

        for _ in range(rnd.randint(0, 4)):
            if len(self.newsfeed) > 0 and rnd.random() < 0.8:
                repost = rnd.choice(self.newsfeed)
                new_msg = Message(
                    mid=f"{repost.mid}_repost_{self.aid}_{self.repost_counter}"
                )
                actions.append(new_msg)
                if logger:
                    logger.debug(
                        f"Agent {self.aid} reposted message {repost.mid} "
                        f"(new ID: {new_msg.mid})"
                    )
                self.repost_counter += 1
            else:
                new_msg = Message(mid=f"{self.aid}_post_{self.post_counter}")
                actions.append(new_msg)
                if logger:
                    logger.debug(f"Agent {self.aid} created new post {new_msg.mid}")
                self.post_counter += 1

        # Newsfeed naive cut-off
        self.newsfeed = self.newsfeed[:cut_off]

        return actions


class NetworkMetrics:
    """Tracks and analyzes network statistics"""

    def __init__(self):
        self.follower_distribution = defaultdict(int)
        self.message_spread = defaultdict(int)
        self.user_activity = defaultdict(int)

    def analyze_network(self, agents):
        """Compute network statistics from current state"""
        self.follower_distribution.clear()
        self.message_spread.clear()
        self.user_activity.clear()

        for agent in agents:
            # Track follower distribution
            follower_count = len(agent.followers)
            self.follower_distribution[follower_count] += 1

            # Track user activity
            total_posts = agent.post_counter + agent.repost_counter
            self.user_activity[agent.aid] = total_posts

            # Track message spread
            for msg in agent.newsfeed:
                self.message_spread[msg.mid] += 1

        # Calculate summary statistics
        total_agents = len(agents)
        total_followers = sum(
            count * num for count, num in self.follower_distribution.items()
        )

        return {
            "total_agents": total_agents,
            "avg_followers": total_followers / total_agents if total_agents > 0 else 0,
            "max_followers": max(self.follower_distribution.keys(), default=0),
            "active_users": sum(1 for v in self.user_activity.values() if v > 0),
            "total_messages": len(self.message_spread),
            "viral_messages": sum(1 for v in self.message_spread.values() if v > 10),
        }


def generate_follower_count(max_followers=100):
    """Generate follower count using power law distribution"""
    alpha = 2.5  # Typical power law exponent for social networks
    x_min = 1
    r = rnd.random()
    # Limit maximum followers to avoid overloading
    return min(int(x_min * (1 - r) ** (-1 / (alpha - 1))), max_followers)


def batch_message_propagation(message_board, agent, messages, logger=None):
    """Efficiently propagate messages to followers with logging"""
    batch_size = 1000
    total_propagations = 0

    for i in range(0, len(agent.followers), batch_size):
        follower_batch = agent.followers[i : i + batch_size]
        for m in messages:
            for follower_aid in follower_batch:
                message_board[follower_aid].append(copy.deepcopy(m))
                total_propagations += 1

    if logger:
        logger.debug(
            f"Propagated {len(messages)} messages to {len(agent.followers)} followers "
            f"(total propagations: {total_propagations})"
        )


def main():
    """Main simulation logic with logging :D"""

    comm_world = MPI.COMM_WORLD
    size = comm_world.Get_size()
    rank = comm_world.Get_rank()

    # Initialize logger only for rank 0 (manager)
    sim_logger = (
        SimulationLogger(verbose=True, log_file="simulation.log")
        if rank == RANK_INDEX["agent_pool_manager"]
        else None
    )

    if size < 2:
        if rank == 0:
            sim_logger.error("Error: This program requires at least 2 processes")
        sys.exit(1)

    if rank == RANK_INDEX["agent_pool_manager"]:
        sim_logger.info("Initializing simulation...")

        # Initialize agent pool
        num_agents = 1000
        agents = [Agent(aid=str(n)) for n in range(num_agents)]

        # Assign followers
        sim_logger.info("Assigning followers to agents...")
        for a in agents:
            num_followers = generate_follower_count()
            followers = rnd.choices(agents, k=num_followers)
            a.followers = [f.aid for f in followers]

        sim_logger.info(f"Initialized {num_agents} agents")

        # Initialize handler management
        ready_handlers_list = list(range(RANK_INDEX["agent_handler"], size))
        busy_handlers_list = []

        # Map agent IDs to messages intended for them
        incoming_message_board = {agent.aid: [] for agent in agents}

        # Start simulation
        comm_world.Barrier()  # wait for bootstrapping
        sim_logger.info("Starting simulation...")

        try:  # if something go wrong: graceful shutdown

            # Just to run some sim with SimSoM xD...
            max_iterations = 1000
            sim_logger.start_progress(max_iterations)
            iteration = 0

            while iteration < max_iterations:

                if agents and ready_handlers_list:

                    # Pick an agent at random
                    agent_index = rnd.choice(range(len(agents)))
                    picked_agent = agents.pop(agent_index)
                    # Pick the first agent handler available
                    handler_rank = ready_handlers_list.pop()

                    sim_logger.debug(
                        f"Assigning agent {picked_agent.aid} to handler {handler_rank}"
                    )

                    # Prepare and send work package
                    agent_pack = (
                        picked_agent,
                        incoming_message_board[picked_agent.aid],
                    )

                    # Flush the incoming messages for this agent
                    incoming_message_board[picked_agent.aid] = []
                    req = comm_world.isend(agent_pack, dest=handler_rank)
                    req.wait()  # wait for non-blocking send accepted (fast)
                    busy_handlers_list.append(handler_rank)

                # Check for completed work.
                # Iterate over a copy of busy handlers list (can be modified).
                for source in busy_handlers_list[:]:

                    status = MPI.Status()

                    if comm_world.Iprobe(source=source, tag=MPI.ANY_TAG, status=status):

                        # Receive and unpack the agent with produced messages
                        agent_pack_reply = comm_world.recv(source=source)
                        updated_agent, new_messages = agent_pack_reply

                        # Re-insert unpacked agent (changed status)
                        agents.append(updated_agent)

                        # Propagate messages through the incoming message board
                        batch_message_propagation(
                            incoming_message_board,
                            updated_agent,
                            new_messages,
                            sim_logger,
                        )

                        # Update handlers status
                        busy_handlers_list.remove(source)
                        ready_handlers_list.append(source)

                        sim_logger.debug(
                            f"Handler {source} completed processing agent "
                            f"{updated_agent.aid}"
                        )

                sim_logger.update_progress()
                iteration += 1

            sim_logger.close_progress()

            # Final metrics
            metrics = NetworkMetrics()
            stats = metrics.analyze_network(agents)

            sim_logger.info("\nSimulation Complete. Network Statistics:")
            for key, value in stats.items():
                sim_logger.info(f"{key}: {value}")

        finally:

            # Ensure to shut down all the agent handler processes
            sim_logger.info("Terminating handler processes...")
            for i in range(RANK_INDEX["agent_handler"], size):
                comm_world.send("sigterm", dest=i)

        # # DEBUG > show agents feeds
        # for a in agents:
        #     print("Agent:", a.aid)
        #     print("Newsfeed:", [m.mid for m in a.newsfeed])

    elif rank >= RANK_INDEX["agent_handler"]:

        # Wait for all the handlers ready
        comm_world.Barrier()

        while True:

            status = MPI.Status()

            # Wait some message from the agent pool manager
            comm = comm_world.recv(
                source=RANK_INDEX["agent_pool_manager"], status=status
            )

            # Quitting case
            if comm == "sigterm":
                break

            # Getting an agent
            agent, messages = comm
            # Add new messsages for this agent on top
            agent.newsfeed = messages + agent.newsfeed
            # Agent scroll the feed and takes some actions
            actions = agent.make_actions()
            # Pack the user (modified status) and actions (messages)
            agent_pack_reply = (agent, actions)
            # Send all back to the agent pool manager and free this handler process
            comm_world.send(agent_pack_reply, dest=RANK_INDEX["agent_pool_manager"])


if __name__ == "__main__":
    main()
