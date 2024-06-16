import itertools
from math import ceil
from typing import TYPE_CHECKING
import networkx as nx

from stream.classes.workload.computation_node import ComputationNode
from zigzag.datatypes import Constants, MemoryOperand

import sys

#from zigzag.hardware.architecture.Core import Core
from stream.classes.hardware.architecture.stream_core import Core

from stream.classes.workload.tensor import Tensor
from stream.classes.hardware.architecture.utils import intersections

if TYPE_CHECKING:
    from stream.classes.hardware.architecture.accelerator import Accelerator


class CommunicationEvent:
    """Represents a communication event involving one or more CommunicationLinks."""

    def __init__(self, id: int, tasks) -> None:
        # Sanity checks
        assert len(tasks) > 0
        assert all([t.type == tasks[0].type] for t in tasks)
        assert all([t.start == tasks[0].start for t in tasks])
        assert all([t.end == tasks[0].end for t in tasks])
        self.id = id
        self.tasks = tasks
        self.type = tasks[0].type
        self.start = tasks[0].start
        self.end = tasks[0].end
        self.energy = sum([t.energy for t in tasks])

    def __str__(self) -> str:
        return f"CommunicationEvent(id={self.id})"

    def __repr__(self) -> str:
        return str(self)


class CommunicationLinkEvent:
    """Represents an event on a communication link.
    An event has:
        - a type, e.g. "transfer" or "block"
        - a start time
        - an end time
        - a list of tensors relevant for the event:
            * the tensor being transferred
            * the tensor(s) for which we are blocking
        - an activity percentage:
            * the percentage of the link bandwidth used
    """

    def __init__(self, type, start, end, tensors, energy, activity=100) -> None:
        self.type = type
        self.start = start
        self.end = end
        self.duration = self.end - self.start
        self.tensors = tensors
        self.energy = energy
        self.activity = activity

    def __str__(self) -> str:
        return f"CommunicationLinkEvent(type={self.type}, start={self.start}, end={self.end}, tensors={self.tensors}, energy={self.energy:.2e}, activity={self.activity:.2f})"

    def __repr__(self) -> str:
        return str(self)

    def get_operands(self):
        return [tensor.layer_operand for tensor in self.tensors]

    def get_origin(self):
        origins = [tensor.origin for tensor in self.tensors]
        assert all([origin == origins[0] for origin in origins])
        return origins[0]


class CommunicationManager:
    """Manages the inter-core and offchip communication of an Accelerator."""

    def __init__(self, accelerator: "Accelerator") -> None:
        self.accelerator = accelerator
        self.shortest_paths = self.get_shortest_paths()
        self.pair_links = self.get_links_for_all_core_pairs()
        self.events = []
        self.event_id = 0

    def get_shortest_paths(self):
        # For each core pair save a shortest path
        shortest_paths = {}
        for producer_core, consumer_core in itertools.product(
            self.accelerator.cores.nodes(), self.accelerator.cores.nodes()
        ):
            shortest_paths[(producer_core, consumer_core)] = nx.shortest_path(
                self.accelerator.cores, producer_core, consumer_core
            )
        return shortest_paths

    def get_links_for_all_core_pairs(self):
        # communication_links = {}
        # for pair, path in self.shortest_paths.items():
        #     traversed_edges = [(i, j) for i, j in zip(path, path[1:])]
        #     communication_links[pair] = [
        #         self.accelerator.cores.edges[traversed_edge]["cl"] for traversed_edge in traversed_edges
        #     ]
        # return communication_links
        # commented the above code to return multiple parallel links instead of one
        cores_pairs = [(producer_core, consumer_core) for producer_core, consumer_core in itertools.product(
            self.accelerator.cores.nodes(), self.accelerator.cores.nodes()
        )]
        communication_links = dict.fromkeys(cores_pairs, []) 
        #print("==== Printing the edges inside get_links_for_all_core_pairs() ====")
        # print(communication_links)
        
        if(self.accelerator.parallel_links_flag == True):
            for producer, consumer, edge_idx in self.accelerator.cores.edges:
                if(communication_links[(producer, consumer)]) == []:
                        communication_links[(producer, consumer)] = [(self.accelerator.cores.edges[(producer, consumer, edge_idx)]["cl"])]#multi_link_cores.edges[(producer, consumer, edge_idx)])] 
                else:
                        communication_links[(producer, consumer)].append(self.accelerator.cores.edges[(producer, consumer, edge_idx)]["cl"])#multi_link_cores.edges[(producer, consumer, edge_idx)])
        else:
            for producer, consumer in self.accelerator.cores.edges:
                if(communication_links[(producer, consumer)]) == []:
                        communication_links[(producer, consumer)] = [(self.accelerator.cores.edges[(producer, consumer)]["cl"])]#multi_link_cores.edges[(producer, consumer, edge_idx)])] 
                else:
                        communication_links[(producer, consumer)].append(self.accelerator.cores.edges[(producer, consumer)]["cl"])#multi_link_cores.edges[(producer, consumer, edge_idx)])
            
        #print(communication_links)
        #print("====================================")
        return communication_links

    def get_links_for_pair(self, sender: Core, receiver: Core):
        """Return the list of traversed CommunicationLinks for sending data from sender core to receiver core.

        Args:
            sender_id (Core): the sending core
            receiver_id (Core): the receiving core
        """
        return self.pair_links[(sender, receiver)]

    def get_links_for_pair_id(self, sender_id: int, receiver_id: int):
        """Return the list of traversed CommunicationLinks for sending data from sender core to receiver core.

        Args:
            sender_id (int): the sending core id
            receiver_id (int): the receiving core id
        """
        # Find the sender and receiver based on the given ids
        sender = self.shortest_paths = self.get_shortest_paths().get_core(sender_id)
        receiver = self.accelerator.get_core(receiver_id)
        return self.get_links_for_pair(sender, receiver)

    def get_all_links(self):
        """Return all unique CommunicationLinks."""
        return list(set(d["cl"] for _, _, d in self.accelerator.cores.edges(data=True)))

    def update_links(
        self,
        tensor: Tensor,
        sender: Core | int,
        receiver: Core | int,
        receiver_memory_operand: str,
        start_timestep: int,
        duration: int,
        chosen_links=None,
    ) -> tuple[int, int, float, float]:
        """Update the links for transfer of a tensor between sender and receiver core at a given timestep.
        A CommunicationEvent is created containing one or more CommunicationLinkEvents,
        i.e. one CommunicationLinkEvent per involved CommunicationLink.

        Args:
            tensor (Tensor): The tensor to be transferred.
            sender (Core): The sending core.
            receiver (Core): The receiving core.
            receiver_memory_operand (str): The memory operand storing the tensor on the receiving end of the transfer.
            start_timestep (int): The timestep at which to start the data transfer.
            duration (int): Duration of the transfer
            chosen_links: Which link was chosen by the get_idle function out of the multiple links that we can choose from.

        Returns:
            int: The timestep at which the transfer is complete.
        """
        end_timestep = start_timestep + duration
        if isinstance(sender, int):
            sender = self.accelerator.get_core(sender)
        if isinstance(receiver, int):
            receiver = self.accelerator.get_core(receiver)
        if not chosen_links:
            links = self.get_links_for_pair(sender, receiver)
        else:
            links = chosen_links
        links = self.get_links_for_pair(sender, receiver)
        if not links:  # When sender == receiver
            return 0, 0

        cles = [
            CommunicationLinkEvent(
                type="transfer",
                start=start_timestep,
                end=end_timestep,
                tensors=[tensor],
                energy=duration * link.unit_energy_cost,
            )
            for link in links
        ]
        event = CommunicationEvent(
            id=self.event_id,
            tasks=cles,
        )
        self.events.append(event)
        self.event_id += 1

        link_energy_cost = 0
        for link, cle in zip(links, cles):
            transfer_energy_cost = link.transfer(cle)
            link_energy_cost += transfer_energy_cost
        # Energy cost of memory reads/writes on sender/receiver
        # For this we need to know the memory operand in order to know where in the sender/receiver the tensor is stored
        # We assume the tensor to be sent is defined from the sender perspective, so we take its operand as the sender memory operand
        sender_memory_operand = tensor.memory_operand
        memory_energy_cost = self.accelerator.get_memory_energy_cost_of_transfer(
            tensor, sender, receiver, sender_memory_operand, receiver_memory_operand
        )
        return link_energy_cost, memory_energy_cost

    def block_offchip_links(
        self,
        too_large_operands: list[MemoryOperand],
        core_id: int,
        start_timestep: int,
        duration: int,
        cn: ComputationNode,
    ) -> int:
        """Block the communication link between 'core' and the offchip core starting at timestep 'start_timestep' for
        duration 'duration'.

        Args:
            too_large_operands (list): List of insufficient memory operands. This decides which links to block
            core_id (int): The core id.
            start_timestep (int): The ideal start timestep of the blocking.
            duration (int): The duration of the blocking in cycles.
            cn (ComputationNode): The computational node for which we are blocking the links.
        """
        links_to_block = dict()
        core = self.accelerator.get_core(core_id)
        offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
        if Constants.OUTPUT_MEM_OP in too_large_operands:
            links_to_offchip = set(self.get_links_for_pair(core, offchip_core))
            req_bw_to_offchip = cn.offchip_bw.wr_in_by_low
            for link in links_to_offchip:
                links_to_block[link] = links_to_block.get(link, 0) + req_bw_to_offchip
        if [op for op in too_large_operands if op != Constants.OUTPUT_MEM_OP]:
            links_from_offchip = set(self.get_links_for_pair(offchip_core, core))
            req_bw_from_offchip = cn.offchip_bw.rd_out_to_low
            for link in links_from_offchip:
                links_to_block[link] = links_to_block.get(link, 0) + req_bw_from_offchip
        if not too_large_operands:
            return start_timestep
        # Get the tensors for which we are blocking based on the operands
        tensors = []
        for mem_op in too_large_operands:
            layer_op = cn.memory_operand_links.mem_to_layer_op(mem_op)
            tensors.append(cn.operand_tensors[layer_op])
        # Get idle window of the involved links
        block_start, new_duration, used_link, all_links_transfer_start_end  = self.get_links_idle_window(links_to_block, start_timestep, duration, tensors)

        for link, req_bw in links_to_block.items():
            req_bw = ceil(req_bw)
            #link.block(block_start, duration, tensors, activity=req_bw)
        used_link.block(block_start, new_duration, tensors, offchip_core, core, activity=req_bw)  # changed it to the duration returned from the get_links_idle_window function
        return block_start

    #def get_links_idle_window(self, links: dict, best_case_start: int, duration: int, tensors: list[Tensor]) -> int:
    def get_links_idle_window(self, links: list, best_case_start: int, tensors: list, duration: int=None, ) -> int:
        """Return the timestep at which tensor can be transfered across the links.
        Both links must have an idle window large enough for the transfer.
        The timestep must be greater than or equal to best_case_start.

        Args:
            links (dict): CommunicationLinks involved in the transfer and their required bandwidth.
            best_case_start (int): The best case start timestep of the transfer.
            duration (int): The required duration of the idle window.
            tensors (list): The tensors to be transferred. Used to broadcast from previous transfer.
        """
        assert len(links) > 0
        idle_intersections = []

        best_idle_intersections = []
        best_idle_intersections.append((sys.maxsize, sys.maxsize))
        best_duration = sys.maxsize

        all_idle_intersections = []

        total_tensors_size = 0
        for t in tensors:
            total_tensors_size += t.size

        # added this to support the potential of having multiple links
        for path in links:
            if hasattr(path, '__iter__'):
                duration = max([ceil(total_tensors_size / link.bandwidth) for link in path])
                for i, (link, req_bw) in enumerate(path.items()):
                    req_bw = min(req_bw, link.bandwidth)  # ceil the bw
                    windows = link.get_idle_window(req_bw, duration, best_case_start, tensors)
                    
                    if i == 0:
                        idle_intersections = windows
                    else:
                        idle_intersections = intersections(idle_intersections, windows)
                        idle_intersections = [
                            period for period in idle_intersections
                            if period[1] - period[0] >= duration
                        ]
                
                all_idle_intersections.append(idle_intersections[0][0]) # contains a copy of all intersections of every path
                
                 #  added this to define a rule for deciding which path to choose
                        #  I'm doing it after the loop since the above loop is meant to go through the multiple links inside one path, in case the cores are not directly connected 
                if windows[0][2] is True:   # favor broadcasting
                    best_idle_intersections = idle_intersections
                    best_duration = duration
                    best_link = path
                else:
                    if idle_intersections[0][0] < best_idle_intersections[0][0]:
                        best_idle_intersections = idle_intersections
                        best_duration = duration
                        best_link = path
            else:
                if not duration:
                    duration = ceil(total_tensors_size / path.bandwidth)
                link = path
                req_bw = path.bandwidth
                req_bw = min(req_bw, link.bandwidth)  # ceil the bw
                windows = link.get_idle_window(req_bw, duration, best_case_start, tensors)
                idle_intersections = windows

                all_idle_intersections.append(idle_intersections[0][0]) # contains a copy of all intersections of every path

                #  added this to define a rule for deciding which path to choose
                        #  I'm doing it after the loop since the above loop is meant to go through the multiple links inside one path, in case the cores are not directly connected 
                if windows[0][2] is True:   # favor broadcasting
                    best_idle_intersections = idle_intersections
                    best_duration = duration
                    best_link = path
                else:
                    if idle_intersections[0][0] < best_idle_intersections[0][0]:
                            best_idle_intersections = idle_intersections
                            best_duration = duration
                            best_link = path

        # Convert the best_link from dict to list of CLs
        if isinstance(best_link, dict):
            best_link = list(best_link.keys())

        return best_idle_intersections[0][0], best_duration, best_link, all_idle_intersections


        # for i, (link, req_bw) in enumerate(links.items()):
        #     req_bw = min(req_bw, link.bandwidth)  # ceil the bw
        #     windows = link.get_idle_window(req_bw, duration, best_case_start, tensors)
        #     if i == 0:
        #         idle_intersections = windows
        #     else:
        #         idle_intersections = intersections(idle_intersections, windows)
        #         idle_intersections = [period for period in idle_intersections if period[1] - period[0] >= duration]
        # return idle_intersections[0][0]
