#!/usr/bin/env python3
"""
GraphRAG with Mayavi 3D Visualization - Enhanced Edition
========================================================
A Graph-based Retrieval-Augmented Generation system using Mayavi for 3D visualization
with improved edge detection and modernized UI
"""

import json
import pickle
import time
import random
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mayavi import mlab
from tvtk.api import tvtk
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
import threading
import atexit
import signal
import sys

@dataclass
class Entity:
    id: str
    name: str
    type: str
    properties: Dict[str, Any]

@dataclass  
class Relation:
    id: str
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]

def preserve_camera(func):
    """Decorator to preserve camera position during operations"""
    def wrapper(self, *args, **kwargs):
        try:
            # Store current camera state
            current_view = mlab.view()
            current_roll = mlab.roll()
            
            # Execute the function
            result = func(self, *args, **kwargs)
            
            # Restore camera state
            mlab.view(*current_view)
            mlab.roll(current_roll)
            
            return result
        except:
            # If there's an error, still try to execute the function
            return func(self, *args, **kwargs)
    return wrapper

class EdgeInfo:
    """Store edge information for picking"""
    def __init__(self, edge, source_pos, target_pos, edge_index, actor):
        self.edge = edge
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.edge_index = edge_index
        self.actor = actor

class GraphRAGMayavi:
    def __init__(self, max_nodes=500, max_edges=500):
        self.graph = nx.MultiDiGraph()
        self.entities = {}
        self.relations = {}
        self.documents = {}
        self.node_positions = None
        self.node_labels = None
        self.text_actors = []  # Store text actors for cleanup
        self.edge_actors = []  # Store edge actors for highlighting
        self.highlighted_edges = []  # Store highlighted edge actors
        self.highlighted_nodes = []  # Store highlighted node actors
        self.edge_info_map = {}  # Map actors to edge info
        self.node_actors = []  # Store node actors
        self.animation_threads = []  # Store animation threads
        self.selected_node = None  # Currently selected node
        self.selected_edge = None  # Currently selected edge
        
        # Configurable limits for performance and visualization
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        print(f"🚀 GraphRAG Enhanced initialized with limits: {max_nodes} nodes, {max_edges} edges")
    
    def set_limits(self, max_nodes=None, max_edges=None):
        """Set or update the visualization limits for nodes and edges"""
        if max_nodes is not None:
            self.max_nodes = max_nodes
            print(f"Updated max nodes limit to: {max_nodes}")
        
        if max_edges is not None:
            self.max_edges = max_edges  
            print(f"Updated max edges limit to: {max_edges}")
        
        print(f"Current limits - Nodes: {self.max_nodes}, Edges: {self.max_edges}")
        
        # If graph already exists, warn about potential need to re-visualize
        if len(self.graph.nodes()) > 0:
            print("⚠️  Note: Changes will take effect on next visualization")
    
    def get_limits(self):
        """Get current visualization limits"""
        return {
            'max_nodes': self.max_nodes,
            'max_edges': self.max_edges,
            'current_nodes': len(self.graph.nodes()),
            'current_edges': len(self.graph.edges())
        }
    
    def get_random_subgraph(self, seed=None):
        """Get a random subset of nodes and edges that connect them"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        all_nodes = list(self.graph.nodes())
        all_edges = list(self.graph.edges())
        
        # Randomly sample nodes
        num_nodes_to_select = min(self.max_nodes, len(all_nodes))
        selected_nodes = random.sample(all_nodes, num_nodes_to_select)
        selected_nodes_set = set(selected_nodes)
        
        print(f"Selected {len(selected_nodes)} random nodes from {len(all_nodes)} total")
        
        # Filter edges to only include those connecting selected nodes
        valid_edges = []
        for edge in all_edges:
            if edge[0] in selected_nodes_set and edge[1] in selected_nodes_set:
                valid_edges.append(edge)
        
        # Randomly sample from valid edges if we have too many
        if len(valid_edges) > self.max_edges:
            selected_edges = random.sample(valid_edges, self.max_edges)
        else:
            selected_edges = valid_edges
        
        print(f"Selected {len(selected_edges)} edges from {len(valid_edges)} valid edges")
        
        return selected_nodes, selected_edges
    
    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, name=entity.name, type=entity.type)
        print(f"Added entity: {entity.name}")
    
    def add_relation(self, relation: Relation):
        self.relations[relation.id] = relation
        self.graph.add_edge(relation.source, relation.target, 
                           relation_type=relation.relation_type)
        print(f"Added relation: {relation.relation_type}")
    
    def query(self, entity_name: str):
        """Query the graph for information about an entity"""
        matches = []
        for eid, entity in self.entities.items():
            if entity_name.lower() in entity.name.lower():
                matches.append(entity)
        return matches
    

    
    def animate_pulse(self, actor, duration=2.0, scale_factor=1.3):
        """Animate a pulsing effect on an actor"""
        try:
            original_scale = actor.actor.scale[0]
            steps = 20
            for i in range(steps):
                if i < steps // 2:
                    # Scale up
                    scale = original_scale + (scale_factor - 1) * original_scale * (i / (steps // 2))
                else:
                    # Scale down
                    scale = scale_factor * original_scale - (scale_factor - 1) * original_scale * ((i - steps // 2) / (steps // 2))
                
                actor.actor.scale = (scale, scale, scale)
                time.sleep(duration / steps)
            
            # Reset to original scale
            actor.actor.scale = (original_scale, original_scale, original_scale)
        except:
            pass
    
    def find_nearest_edge(self, click_pos):
        """Find the nearest edge to a given position"""
        try:
            min_distance = float('inf')
            nearest_edge_info = None
            
            # Convert 2D click position to 3D world coordinates if needed
            # For now, we'll use the picked position from the picker
            
            for edge_info in self.edge_info_map.values():
                # Calculate distance from click position to edge midpoint
                mid_x = (edge_info.source_pos[0] + edge_info.target_pos[0]) / 2
                mid_y = (edge_info.source_pos[1] + edge_info.target_pos[1]) / 2
                mid_z = (edge_info.source_pos[2] + edge_info.target_pos[2]) / 2
                
                # Calculate distance to midpoint
                distance = np.sqrt((mid_x - click_pos[0])**2 + 
                                 (mid_y - click_pos[1])**2 + 
                                 (mid_z - click_pos[2])**2)
                
                # Also calculate minimum distance to the line segment
                line_distance = self.point_to_line_distance(click_pos, edge_info.source_pos, edge_info.target_pos)
                
                # Use the minimum of midpoint distance and line distance
                edge_distance = min(distance, line_distance)
                
                if edge_distance < min_distance:
                    min_distance = edge_distance
                    nearest_edge_info = edge_info
            
            return nearest_edge_info, min_distance
            
        except Exception as e:
            print(f"Error finding nearest edge: {e}")
            return None, float('inf')
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate the distance from a point to a line segment"""
        try:
            # Convert to numpy arrays
            p = np.array(point)
            a = np.array(line_start)
            b = np.array(line_end)
            
            # Vector from line start to end
            ab = b - a
            # Vector from line start to point
            ap = p - a
            
            # Handle degenerate case (point edge)
            ab_squared = np.dot(ab, ab)
            if ab_squared < 1e-10:
                return np.linalg.norm(ap)
            
            # Project point onto line (parameter t)
            t = np.dot(ap, ab) / ab_squared
            
            # Clamp t to [0, 1] to stay within line segment
            t = max(0.0, min(1.0, t))
            
            # Find closest point on line segment
            projection = a + t * ab
            
            # Calculate distance
            distance = np.linalg.norm(p - projection)
            
            return distance
            
        except Exception as e:
            print(f"Error in point-to-line distance calculation: {e}")
            return float('inf')
    
    def clear_all_text(self):
        """Manually clear all text information from the screen"""
        try:
            for actor in self.text_actors:
                try:
                    actor.remove()
                except:
                    pass
            self.text_actors = []
            print("✓ Cleared all text information")
        except Exception as e:
            print(f"Error clearing text: {e}")
    
    def on_mouse_click(self, obj, event):
        """Handle mouse click events with improved edge detection"""
        try:
            # Get the interactor
            interactor = obj
            
            # Check if control key is pressed
            ctrl_pressed = interactor.GetControlKey()
            shift_pressed = interactor.GetShiftKey()
            
            # Get the click position
            click_pos = interactor.GetEventPosition()
            
            # Create a picker
            picker = tvtk.CellPicker()
            picker.tolerance = 0.01
            
            # Pick at the click position
            renderer = interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()
            picker.pick(click_pos[0], click_pos[1], 0, renderer)
            
            # Get the picked actor and position
            picked_actor = picker.actor
            picked_pos = picker.pick_position
            
            if picked_actor:
                # Check if we picked an edge
                edge_found = False
                for actor_id, edge_info in self.edge_info_map.items():
                    if edge_info.actor and edge_info.actor.actor.actor == picked_actor:
                        if ctrl_pressed:
                            self.highlight_edge_enhanced(edge_info, picked_pos)
                        else:
                            self.show_edge_info_panel(edge_info)
                        edge_found = True
                        break
                
                if not edge_found:
                    # Check for nodes
                    closest_entity = None
                    min_node_distance = float('inf')
                    
                    if self.node_positions is not None:
                        for i, (x, y, z) in enumerate(self.node_positions):
                            distance = np.sqrt((x - picked_pos[0])**2 + 
                                             (y - picked_pos[1])**2 + 
                                             (z - picked_pos[2])**2)
                            if distance < min_node_distance:
                                min_node_distance = distance
                                closest_entity = i
                    
                    # If click is close to a node, show entity info
                    if closest_entity is not None and min_node_distance < 0.5:
                        if shift_pressed:
                            self.show_node_connections(closest_entity)
                        else:
                            self.display_entity_info_enhanced(closest_entity, picked_pos)
                    elif shift_pressed:
                        # Shift-click on empty space: highlight nearest edge
                        nearest_edge, distance = self.find_nearest_edge(picked_pos)
                        if nearest_edge:
                            print(f"\n🎯 Highlighting nearest edge (distance: {distance:.3f})")
                            self.highlight_edge_enhanced(nearest_edge, picked_pos)
            else:
                if shift_pressed and picked_pos and any(pos != 0 for pos in picked_pos):
                    # Shift-click on empty space: highlight nearest edge
                    nearest_edge, distance = self.find_nearest_edge(picked_pos)
                    if nearest_edge:
                        print(f"\n🎯 Highlighting nearest edge (distance: {distance:.3f})")
                        self.highlight_edge_enhanced(nearest_edge, picked_pos)
                else:
                    # Clear selection if clicking on empty space without shift
                    self.clear_all_highlights()
                    
        except Exception as e:
            print(f"Click handling error: {e}")
            import traceback
            traceback.print_exc()
    
    @preserve_camera
    def clear_all_highlights(self):
        """Clear all highlights and selections but keep text information visible"""
        try:
            # Clear edge highlights
            for actor in self.highlighted_edges:
                try:
                    actor.remove()
                except:
                    pass
            self.highlighted_edges = []
            
            # Clear node highlights
            for actor in self.highlighted_nodes:
                try:
                    actor.remove()
                except:
                    pass
            self.highlighted_nodes = []
            
            # NOTE: Text actors are NOT cleared - they stay visible
            # Only clear them when explicitly requested
            
            self.selected_node = None
            self.selected_edge = None
            
        except Exception as e:
            print(f"Error clearing highlights: {e}")
    
    @preserve_camera
    def show_node_connections(self, node_index):
        """Show all connections for a node"""
        try:
            if node_index >= len(self.node_labels):
                return
            
            node_id = None
            node_name = self.node_labels[node_index]
            
            # Find the node ID
            for eid, entity in self.entities.items():
                if hasattr(entity, 'name') and entity.name == node_name:
                    node_id = eid
                    break
            
            if not node_id:
                return
            
            # Clear previous highlights
            self.clear_all_highlights()
            
            # Highlight the node
            pos = self.node_positions[node_index]
            self.highlight_node_enhanced(node_index, pos)
            
            # Highlight all connected edges
            connected_edges = []
            for edge in self.graph.edges():
                if edge[0] == node_id or edge[1] == node_id:
                    connected_edges.append(edge)
            
            # Highlight each connected edge
            for edge_info in self.edge_info_map.values():
                if edge_info.edge in connected_edges:
                    self.highlight_edge_simple(edge_info)
            
            print(f"\n🔗 Showing {len(connected_edges)} connections for: {node_name}")
            
        except Exception as e:
            print(f"Error showing node connections: {e}")
    
    @preserve_camera
    def highlight_edge_simple(self, edge_info):
        """Simple edge highlighting without animation"""
        try:
            # Create highlighted edge
            highlighted_actor = mlab.plot3d(
                [edge_info.source_pos[0], edge_info.target_pos[0]],
                [edge_info.source_pos[1], edge_info.target_pos[1]], 
                [edge_info.source_pos[2], edge_info.target_pos[2]],
                color=(1, 0.7, 0.2),  # Golden highlight
                opacity=0.9,
                tube_radius=0.05
            )
            
            self.highlighted_edges.append(highlighted_actor)
            
        except Exception as e:
            print(f"Error highlighting edge: {e}")
    
    @preserve_camera
    def show_edge_info_panel(self, edge_info):
        """Show detailed edge information in a panel"""
        try:
            # Get edge information
            source_entity = self.entities.get(edge_info.edge[0])
            target_entity = self.entities.get(edge_info.edge[1])
            
            source_name = source_entity.name if source_entity else str(edge_info.edge[0])
            target_name = target_entity.name if target_entity else str(edge_info.edge[1])
            
            # Get edge data
            edge_data = self.graph.get_edge_data(edge_info.edge[0], edge_info.edge[1])
            relation_type = "RELATED"
            if edge_data:
                for key, data in edge_data.items():
                    if 'relation_type' in data:
                        relation_type = data['relation_type']
                        break
            
            # Create info panel text
            info_text = f"━━━ EDGE INFORMATION ━━━\n"
            info_text += f"From: {source_name}\n"
            info_text += f"To: {target_name}\n"
            info_text += f"Type: {relation_type}\n"
            info_text += f"━━━━━━━━━━━━━━━━━━━"
            
            # Display in 3D space
            mid_pos = [(edge_info.source_pos[i] + edge_info.target_pos[i]) / 2 for i in range(3)]
            
            # Don't clear previous text - let info accumulate
            
            # Add new text
            text_actor = mlab.text3d(
                mid_pos[0], mid_pos[1], mid_pos[2] + 0.3,
                info_text, 
                scale=0.02, 
                color=(0.9, 0.9, 1.0)
            )
            self.text_actors.append(text_actor)
            
            # Also print to console
            print(f"\n{info_text}")
            
            # Highlight the edge
            self.highlight_edge_enhanced(edge_info, mid_pos)
            
        except Exception as e:
            print(f"Error showing edge info panel: {e}")
    
    @preserve_camera
    def highlight_edge_enhanced(self, edge_info, picked_pos):
        """Enhanced edge highlighting with modern effects"""
        try:
            # Clear previous highlights
            for actor in self.highlighted_edges:
                try:
                    actor.remove()
                except:
                    pass
            self.highlighted_edges = []
            
            # Create main highlighted edge with animated effect
            # Main highlight
            highlighted_actor = mlab.plot3d(
                [edge_info.source_pos[0], edge_info.target_pos[0]],
                [edge_info.source_pos[1], edge_info.target_pos[1]], 
                [edge_info.source_pos[2], edge_info.target_pos[2]],
                color=(1, 0.3, 0.1),  # Bright orange-red
                opacity=0.95,
                tube_radius=0.06
            )
            
            # Add glow effect
            glow_actor = mlab.plot3d(
                [edge_info.source_pos[0], edge_info.target_pos[0]],
                [edge_info.source_pos[1], edge_info.target_pos[1]], 
                [edge_info.source_pos[2], edge_info.target_pos[2]],
                color=(1, 0.5, 0.2),  # Orange glow
                opacity=0.3,
                tube_radius=0.12  # Larger for glow effect
            )
            
            self.highlighted_edges.extend([highlighted_actor, glow_actor])
            
            # Animate the highlight
            def animate():
                self.animate_pulse(highlighted_actor, duration=1.0, scale_factor=1.2)
            
            thread = threading.Thread(target=animate, daemon=True)
            thread.start()
            self.animation_threads.append(thread)
            
            # Get edge information for display
            source_entity = self.entities.get(edge_info.edge[0])
            target_entity = self.entities.get(edge_info.edge[1])
            
            source_name = source_entity.name if source_entity else str(edge_info.edge[0])
            target_name = target_entity.name if target_entity else str(edge_info.edge[1])
            
            # Create floating label with modern styling
            mid_pos = [(edge_info.source_pos[i] + edge_info.target_pos[i]) / 2 for i in range(3)]
            edge_label = f"🔗 {source_name} → {target_name}"
            
            text_actor = mlab.text3d(
                mid_pos[0], mid_pos[1], mid_pos[2] + 0.2, 
                edge_label, 
                scale=0.035, 
                color=(1, 0.9, 0.6)
            )
            # Bold is not supported in Mayavi, using larger size instead
            self.text_actors.append(text_actor)
            
            self.selected_edge = edge_info
            
        except Exception as e:
            print(f"Error in enhanced edge highlighting: {e}")
    
    @preserve_camera
    def highlight_node_enhanced(self, node_index, position):
        """Enhanced node highlighting with modern effects"""
        try:
            # Clear previous node highlights
            for actor in self.highlighted_nodes:
                try:
                    actor.remove()
                except:
                    pass
            self.highlighted_nodes = []
            
            # Create multi-layer highlight effect
            # Core highlight
            core_highlight = mlab.points3d(
                [position[0]], [position[1]], [position[2]],
                [1],
                scale_mode='none',
                scale_factor=0.18,
                color=(1, 0.2, 0.2),  # Bright red
                opacity=1.0,
                resolution=32
            )
            
            # Inner glow
            inner_glow = mlab.points3d(
                [position[0]], [position[1]], [position[2]],
                [1],
                scale_mode='none',
                scale_factor=0.25,
                color=(1, 0.4, 0.4),
                opacity=0.6,
                resolution=24
            )
            
            # Outer glow
            outer_glow = mlab.points3d(
                [position[0]], [position[1]], [position[2]],
                [1],
                scale_mode='none',
                scale_factor=0.35,
                color=(1, 0.6, 0.6),
                opacity=0.3,
                resolution=16
            )
            
            # Add ripple effect
            ripple = mlab.points3d(
                [position[0]], [position[1]], [position[2]],
                [1],
                scale_mode='none',
                scale_factor=0.15,
                color=(1, 1, 0.8),
                opacity=0.8,
                resolution=32
            )
            
            self.highlighted_nodes.extend([core_highlight, inner_glow, outer_glow, ripple])
            
            # Animate ripple effect
            def animate_ripple():
                try:
                    for i in range(20):
                        scale = 0.15 + i * 0.02
                        opacity = 0.8 - i * 0.04
                        ripple.actor.scale = (scale, scale, scale)
                        ripple.actor.property.opacity = max(0, opacity)
                        time.sleep(0.05)
                    ripple.actor.property.opacity = 0
                except:
                    pass
            
            thread = threading.Thread(target=animate_ripple, daemon=True)
            thread.start()
            self.animation_threads.append(thread)
            
            self.selected_node = node_index
            
        except Exception as e:
            print(f"Error in enhanced node highlighting: {e}")
    
    @preserve_camera
    def display_entity_info_enhanced(self, entity_index, position):
        """Enhanced entity information display with modern UI"""
        try:
            # Don't clear previous text - let info accumulate on screen
            
            # Highlight the node with enhanced effects
            self.highlight_node_enhanced(entity_index, position)
            
            # Get entity information
            if entity_index >= len(self.node_labels):
                return
            
            entity_name = self.node_labels[entity_index]
            entity = None
            
            for ent in self.entities.values():
                if hasattr(ent, 'name') and ent.name == entity_name:
                    entity = ent
                    break
            
            # Prepare entity information
            name = entity_name if entity_name else "Unknown"
            entity_type = "Unknown"
            if entity and hasattr(entity, 'type') and entity.type:
                entity_type = str(entity.type)
            
            # Create modern info card
            info_text = f"ENTITY INFO\nName: {name[:25]}\nType: {entity_type}"
            
            # Position the text above and to the right of the node
            text_pos_x = position[0] + 0.3
            text_pos_y = position[1] + 0.3
            text_pos_z = position[2] + 0.3
            
            # Create main info text (removed shadow for clarity)
            text_actor = mlab.text3d(
                text_pos_x, text_pos_y, text_pos_z,
                info_text,
                scale=0.025,
                color=(1, 0.95, 0.8)
            )
            # Bold is not supported in Mayavi, using bright color instead
            self.text_actors.append(text_actor)
            
            # Add connection count
            degree = self.graph.degree(list(self.graph.nodes())[entity_index]) if entity_index < len(self.graph.nodes()) else 0
            conn_text = f"⚡ {degree} connections"
            conn_actor = mlab.text3d(
                position[0], position[1], position[2] - 0.2,
                conn_text,
                scale=0.02,
                color=(0.8, 0.8, 1.0)
            )
            self.text_actors.append(conn_actor)
            
            # Print to console with formatting
            print(f"\n{'═' * 40}")
            print(f"🎯 SELECTED ENTITY")
            print(f"{'─' * 40}")
            print(f"  Name: {name}")
            print(f"  Type: {entity_type}")
            print(f"  Connections: {degree}")
            print(f"{'═' * 40}")
            
            # Info will stay visible until cleared manually or another entity is selected
            
        except Exception as e:
            print(f"Error displaying enhanced entity info: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_mayavi_3d_interactive(self, save_path=None, seed=None):
        """Enhanced 3D visualization with modern UI and improved edge detection"""
        if len(self.graph.nodes()) == 0:
            print("No nodes to visualize")
            return

        print("\n🌟 Creating Enhanced Interactive 3D Visualization...")
        print("━" * 50)
        print("🎮 CONTROLS:")
        print("  • LEFT CLICK on nodes → Show entity details (stays visible)")
        print("  • LEFT CLICK on edges → Show connection info (stays visible)")
        print("  • SHIFT+CLICK on nodes → Highlight all connections")
        print("  • SHIFT+CLICK anywhere → Highlight nearest edge")
        print("  • CTRL+CLICK on edges → Enhanced edge highlight")
        print("  • Click empty space → Clear highlights (text stays)")
        print("  • Press 'C' → Clear all text information")
        print("  • Mouse drag → Rotate view")
        print("  • Mouse wheel → Zoom in/out")
        print("━" * 50)

        # Get random subset of nodes and edges
        selected_nodes, selected_edges = self.get_random_subgraph(seed=seed)
        
        # Create subgraph for layout calculation
        subgraph = self.graph.subgraph(selected_nodes)

        # Clear any existing plots and data
        mlab.clf()
        self.edge_info_map.clear()
        self.node_actors.clear()
        self.text_actors.clear()
        self.highlighted_edges.clear()
        self.highlighted_nodes.clear()
        
        # Set up modern dark theme
        fig = mlab.figure(
            bgcolor=(0.05, 0.05, 0.08),  # Very dark blue-gray
            fgcolor=(0.9, 0.9, 0.95),
            size=(1600, 1000)
        )

        # Get 3D positions using force-directed layout
        pos_2d = nx.spring_layout(subgraph, k=5, iterations=100, scale=3.5, seed=seed)

        # Convert to 3D positions with better distribution
        positions = []
        node_colors = []
        node_sizes = []
        node_labels = []

        # Modern color palette for entity types
        modern_colors = {
            'PERSON': (0.96, 0.26, 0.21),      # Material Red
            'ORGANIZATION': (0.13, 0.59, 0.95), # Material Blue
            'LOCATION': (0.30, 0.69, 0.31),     # Material Green
            'CONCEPT': (1.00, 0.60, 0.00),      # Material Orange
            'EVENT': (0.61, 0.15, 0.69),        # Material Purple
            'ENTITY': (1.00, 0.92, 0.23),       # Material Yellow
            'UNKNOWN': (0.62, 0.62, 0.62)       # Material Gray
        }

        # Process nodes
        for i, (node, (x, y)) in enumerate(pos_2d.items()):
            # Create 3D position with better Z distribution
            z = np.sin(i * 0.5) * 2.0 if len(subgraph.nodes()) > 1 else 0
            positions.append([x, y, z])

            # Get entity for styling
            entity = self.entities.get(node, Entity('', '', 'UNKNOWN', {}))
            entity_name = entity.name if hasattr(entity, 'name') else str(node)
            entity_type = entity.type if hasattr(entity, 'type') else 'UNKNOWN'
            
            # Use gradient coloring based on connections
            degree = subgraph.degree(node)
            color_intensity = min(1.0, 0.3 + (degree * 0.1))
            node_colors.append(color_intensity * 100)
            
            # Dynamic node sizing
            base_size = 0.1
            size_factor = 1.0 + (degree * 0.05)
            node_sizes.append(min(0.3, base_size * size_factor))
            
            node_labels.append(entity_name)

        # Convert to numpy arrays
        positions = np.array(positions)
        node_colors = np.array(node_colors)
        node_sizes = np.array(node_sizes)

        # Store for interaction
        self.node_positions = positions
        self.node_labels = node_labels

        # Create nodes with modern styling
        pts = mlab.points3d(
            positions[:, 0], 
            positions[:, 1], 
            positions[:, 2],
            node_colors,
            scale_mode='none',
            scale_factor=0.12,
            colormap='viridis',  # Modern colormap
            opacity=0.95,
            resolution=24
        )
        
        # Enhanced node appearance
        pts.actor.property.specular = 0.4
        pts.actor.property.specular_power = 80
        pts.actor.property.ambient = 0.2
        pts.actor.property.diffuse = 0.8
        pts.actor.property.interpolation = 'phong'

        # Create edges with improved detection
        edge_count = 0
        print(f"\n📊 Rendering {len(positions)} nodes and {len(selected_edges)} edges...")
        
        for edge in selected_edges:
            try:
                source_idx = selected_nodes.index(edge[0])
                target_idx = selected_nodes.index(edge[1])

                source_pos = positions[source_idx]
                target_pos = positions[target_idx]

                # Create visible edge with simple color
                visible_edge = mlab.plot3d(
                    [source_pos[0], target_pos[0]],
                    [source_pos[1], target_pos[1]], 
                    [source_pos[2], target_pos[2]],
                    color=(0.3, 0.5, 0.8),  # Nice blue color
                    opacity=0.7,
                    tube_radius=0.025
                )
                
                # Create larger invisible collision detection tube
                collision_tube = mlab.plot3d(
                    [source_pos[0], target_pos[0]],
                    [source_pos[1], target_pos[1]], 
                    [source_pos[2], target_pos[2]],
                    color=(1, 1, 1),
                    opacity=0.0,  # Completely invisible
                    tube_radius=0.08  # Larger for easier clicking
                )
                
                # Store edge information properly
                edge_info = EdgeInfo(edge, source_pos, target_pos, edge_count, collision_tube)
                actor_id = id(collision_tube.actor.actor)
                self.edge_info_map[actor_id] = edge_info
                
                # Also store for the visible edge
                visible_actor_id = id(visible_edge.actor.actor)
                self.edge_info_map[visible_actor_id] = edge_info
                
                edge_count += 1
                
            except Exception as e:
                print(f"Error drawing edge {edge}: {e}")
                continue

        # Add modern UI elements
        # Title with gradient effect
        title_text = '🌐 Knowledge Graph Explorer'
        mlab.title(title_text, color=(0.95, 0.95, 1.0), size=0.12, height=0.96)

        # Add stats panel
        stats_text = f"Nodes: {len(positions)} Edges: {edge_count} Density: {nx.density(subgraph):.3f}"
        mlab.text(0.02, 0.02, stats_text, color=(0.7, 0.8, 0.9), width=0.3)

        # Enhanced lighting for modern look
        scene = mlab.gcf().scene
        
        # Add multiple light sources for better illumination
        try:
            light_manager = scene.light_manager
            if light_manager and hasattr(light_manager, 'lights') and len(light_manager.lights) > 0:
                # Main light
                light1 = light_manager.lights[0]
                light1.intensity = 0.7
                light1.position = (1, 1, 1)
                
                # Add rim lighting if possible
                if hasattr(light_manager, 'add_light'):
                    rim_light = light_manager.add_light()
                    rim_light.intensity = 0.3
                    rim_light.position = (-1, -1, 0.5)
        except Exception as e:
            print(f"Note: Could not adjust lighting: {e}")
        
        # Set up enhanced interaction
        try:
            figure = mlab.gcf()
            interactor = figure.scene.interactor
            
            # Remove any existing observers to avoid duplicates
            interactor.remove_observers('LeftButtonPressEvent')
            interactor.remove_observers('KeyPressEvent')
            
            # Add event observer for mouse clicks
            interactor.add_observer('LeftButtonPressEvent', self.on_mouse_click)
            
            # Add keyboard event observer
            def on_key_press(obj, event):
                key = obj.GetKeySym()
                if key.lower() == 'c':
                    self.clear_all_text()
            
            interactor.add_observer('KeyPressEvent', on_key_press)
            
            print("✅ Enhanced interaction system activated!")
            print("💡 Press 'C' to clear all text information")
            
        except Exception as e:
            print(f"⚠️  Error setting up interaction: {e}")
            import traceback
            traceback.print_exc()

        # Set initial camera position for best view
        mlab.view(azimuth=45, elevation=65, distance='auto', focalpoint='auto')
        
        # Enable anti-aliasing for smoother visuals
        scene.anti_aliasing_frames = 8

        if save_path:
            try:
                # High quality save
                mlab.savefig(save_path, size=(2400, 1600), figure=fig)
                print(f"✅ High-resolution visualization saved to {save_path}")
            except Exception as e:
                print(f"❌ Error saving image: {e}")

        # Show the interactive plot
        print("\n🚀 Visualization ready! Close the window when done.")
        
        # Set up cleanup to close window when terminal closes
        import atexit
        import signal
        
        def cleanup():
            try:
                mlab.close(all=True)
            except:
                pass
        
        # Register cleanup for normal exit
        atexit.register(cleanup)
        
        # Handle terminal signals (Ctrl+C, terminal close, etc.)
        def signal_handler(signum, frame):
            cleanup()
            import sys
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            mlab.show()
        except KeyboardInterrupt:
            cleanup()
        finally:
            cleanup()

    def simple_picker_callback(self, picker_obj):
        """Simplified picker callback for compatibility"""
        pass  # Keeping for backward compatibility
    
    def visualize_mayavi_3d(self, save_path=None):
        """Visualize the knowledge graph in 3D using Mayavi (non-interactive version)"""
        if len(self.graph.nodes()) == 0:
            print("No nodes to visualize")
            return
        
        print("Creating Mayavi 3D visualization...")
        
        # Clear any existing plots
        mlab.clf()
        
        # Get 3D positions using NetworkX layout with increased scale
        pos_2d = nx.spring_layout(self.graph, k=5, iterations=100, scale=3.0)
        
        # Convert to 3D positions and prepare data
        positions = []
        node_colors = []
        node_sizes = []
        node_labels = []
        
        # Define colors for different entity types
        type_colors = {
            'PERSON': 0,      # Red
            'ORGANIZATION': 1, # Blue 
            'LOCATION': 2,     # Green
            'CONCEPT': 3,      # Orange
            'EVENT': 4,        # Purple
            'ENTITY': 5,       # Yellow
            'UNKNOWN': 6       # Gray
        }
        
        for i, (node, (x, y)) in enumerate(pos_2d.items()):
            # Add more spread to Z coordinate for better 3D visualization with increased scale
            z = np.random.uniform(-2.5, 2.5) if len(self.graph.nodes()) > 1 else 0
            positions.append([x, y, z])
            
            # Get entity type for coloring
            entity = self.entities.get(node, Entity('', '', 'UNKNOWN', {}))
            entity_type = entity.type
            color_index = type_colors.get(entity_type, 6)
            node_colors.append(color_index)
            
            # Node size based on degree (connections)
            degree = self.graph.degree(node)
            node_sizes.append(max(0.05, degree * 0.02))
            
            # Store label
            node_labels.append(entity.name[:20])  # Truncate long names
        
        # Convert to numpy arrays
        positions = np.array(positions)
        node_colors = np.array(node_colors)
        node_sizes = np.array(node_sizes)
        
        # Plot nodes as 3D points
        pts = mlab.points3d(
            positions[:, 0], 
            positions[:, 1], 
            positions[:, 2],
            node_colors,
            scale_mode='none',
            scale_factor=0.15,  # Reasonable size
            colormap='Set1',
            opacity=0.8
        )
        
        # Add edges as lines
        for edge in self.graph.edges():
            source_idx = list(self.graph.nodes()).index(edge[0])
            target_idx = list(self.graph.nodes()).index(edge[1])
            
            source_pos = positions[source_idx]
            target_pos = positions[target_idx]
            
            # Create line between nodes
            mlab.plot3d(
                [source_pos[0], target_pos[0]],
                [source_pos[1], target_pos[1]], 
                [source_pos[2], target_pos[2]],
                color=(0.5, 0.5, 0.7),
                opacity=0.5,
                tube_radius=0.025  # Reasonable thickness
            )
        
        # Add title
        mlab.title('Knowledge Graph 3D Visualization (Mayavi)', size=0.3)
        
        # Add axes
        mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
        
        # Add colorbar
        mlab.colorbar(pts, title="Entity Types", orientation="vertical")
        
        # Improve visualization
        mlab.view(azimuth=45, elevation=60, distance='auto')
        
        if save_path:
            mlab.savefig(save_path, size=(1200, 800))
            print(f"Mayavi 3D visualization saved to {save_path}")
        
        # Show the plot
        mlab.show()
    
    def get_statistics(self):
        """Get graph statistics"""
        return {
            'nodes': len(self.graph.nodes()),
            'edges': len(self.graph.edges()),
            'entities': len(self.entities),
            'relations': len(self.relations),
            'density': nx.density(self.graph) if len(self.graph.nodes()) > 1 else 0,
            'is_connected': nx.is_connected(self.graph.to_undirected()) if len(self.graph.nodes()) > 1 else True
        }
    
    def load_from_knowledge_graph_pkl(self, filepath="knowledge_graph.pkl"):
        """Load from the specific knowledge graph pickle format"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded {filepath}")
            
            entity_map = data.get('entity_map', {})
            relationship_map = data.get('relationship_map', {})
            
            print(f"Found {len(entity_map)} entities and {len(relationship_map)} relationship entries")
            
            # Convert entities (sample for performance)
            max_entities = self.max_nodes  # Use configurable limit
            entity_count = 0
            
            print(f"Loading up to {max_entities} entities...")
            
            for entity_name, entity_info in list(entity_map.items())[:max_entities]:
                if isinstance(entity_info, dict):
                    entity = Entity(
                        id=f"entity_{entity_count}",
                        name=entity_name,
                        type=entity_info.get('type', 'ENTITY'),
                        properties=entity_info
                    )
                    self.add_entity(entity)
                    entity_count += 1
            
            # Convert relationships
            entity_name_to_id = {entity.name: entity.id for entity in self.entities.values()}
            relation_count = 0
            max_relations = self.max_edges  # Use configurable limit
            
            print(f"Loading up to {max_relations} relationships...")
            
            for source_entity, relationships in relationship_map.items():
                if relation_count >= max_relations:
                    break
                    
                if source_entity in entity_name_to_id:
                    source_id = entity_name_to_id[source_entity]
                    
                    if isinstance(relationships, list):
                        for rel_info in relationships:
                            if relation_count >= max_relations:
                                break
                                
                            if isinstance(rel_info, dict):
                                target_entity = rel_info.get('target', '')
                                relation_type = rel_info.get('type', 'RELATED')
                                
                                if target_entity in entity_name_to_id:
                                    target_id = entity_name_to_id[target_entity]
                                    
                                    relation = Relation(
                                        id=f"rel_{relation_count}",
                                        source=source_id,
                                        target=target_id,
                                        relation_type=relation_type,
                                        properties=rel_info
                                    )
                                    self.add_relation(relation)
                                    relation_count += 1
            
            print(f"✓ Loaded {entity_count} entities and {relation_count} relations")
            return True
            
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return False

    def visualize_2d_interactive(self, save_path=None, seed=None, figsize=(12, 8)):
        """Visualize the knowledge graph in 2D using matplotlib with random sampling"""
        if len(self.graph.nodes()) == 0:
            print("No nodes to visualize")
            return

        print("Creating 2D visualization...")
        
        # Get random subset of nodes and edges
        selected_nodes, selected_edges = self.get_random_subgraph(seed=seed)
        
        # Create subgraph for layout calculation
        subgraph = self.graph.subgraph(selected_nodes)
        
        # Calculate 2D layout with increased scale
        pos = nx.spring_layout(subgraph, k=5, iterations=100, scale=2.5)
        
        # Create figure
        plt.figure(figsize=figsize)
        plt.clf()
        
        # Set dark background
        plt.style.use('dark_background')
        fig = plt.gcf()
        fig.patch.set_facecolor('black')
        
        # Prepare node data
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        # Define colors for different entity types
        type_colors = {
            'PERSON': '#FF6B6B',      # Red
            'ORGANIZATION': '#4ECDC4', # Teal 
            'LOCATION': '#45B7D1',     # Blue
            'CONCEPT': '#FFA07A',      # Orange
            'EVENT': '#98D8C8',        # Green
            'ENTITY': '#F7DC6F',       # Yellow
            'UNKNOWN': '#BDC3C7'       # Gray
        }
        
        for node in selected_nodes:
            # Get entity for coloring
            entity = self.entities.get(node, Entity('', '', 'UNKNOWN', {}))
            entity_type = entity.type
            color = type_colors.get(entity_type, type_colors['UNKNOWN'])
            node_colors.append(color)
            
            # Node size based on degree
            degree = subgraph.degree(node)
            node_sizes.append(max(100, degree * 50))
            
            # Store label
            entity_name = entity.name if hasattr(entity, 'name') else str(node)
            node_labels[node] = entity_name[:15] + "..." if len(entity_name) > 15 else entity_name
        
        # Draw edges first (so they appear behind nodes)
        edge_list = [(edge[0], edge[1]) for edge in selected_edges]
        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=edge_list,
            edge_color='#34495E',
            alpha=0.6,
            width=1.5
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            edgecolors='white',
            linewidths=1
        )
        
        # Add labels
        nx.draw_networkx_labels(
            subgraph, pos,
            labels=node_labels,
            font_size=8,
            font_color='white',
            font_weight='bold'
        )
        
        # Customize plot
        plt.title(f'Knowledge Graph 2D Visualization\n{len(selected_nodes)} nodes, {len(selected_edges)} edges', 
                 color='white', fontsize=16, pad=20)
        plt.axis('off')
        
        # Add legend for entity types
        legend_elements = []
        used_types = set()
        for node in selected_nodes:
            entity = self.entities.get(node, Entity('', '', 'UNKNOWN', {}))
            if entity.type not in used_types:
                used_types.add(entity.type)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=type_colors.get(entity.type, type_colors['UNKNOWN']), 
                                                markersize=8, label=entity.type, linestyle='None'))
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right', 
                      framealpha=0.8, facecolor='black', edgecolor='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            print(f"2D visualization saved to {save_path}")
        
        plt.show()
        
        print(f"✓ 2D visualization completed with {len(selected_nodes)} nodes and {len(selected_edges)} edges")

def demo_mayavi_knowledge_graph(max_nodes=500, max_edges=500, mode='3d'):
    """Demo using Mayavi to visualize the knowledge graph with click functionality"""
    print("GraphRAG with Interactive Visualization Demo")
    print("=" * 45)
    
    # Initialize GraphRAG with Mayavi and custom limits
    graph_rag = GraphRAGMayavi(max_nodes=max_nodes, max_edges=max_edges)
    
    # Load the knowledge graph
    print("Loading knowledge_graph.pkl...")
    success = graph_rag.load_from_knowledge_graph_pkl()
    
    if not success:
        print("Failed to load knowledge graph. Creating sample data instead...")
        # Create sample data
        entities = [
            Entity("person_1", "John Smith", "PERSON", {"age": 30, "department": "Engineering"}),
            Entity("company_1", "Tech Corp", "ORGANIZATION", {"industry": "Technology", "employees": 500}),
            Entity("location_1", "San Francisco", "LOCATION", {"state": "California", "population": 875000}),
            Entity("project_1", "AI Project", "CONCEPT", {"status": "active", "budget": 1000000}),
            Entity("event_1", "Conference 2024", "EVENT", {"date": "2024-03-15", "attendees": 1000})
        ]
        
        for entity in entities:
            graph_rag.add_entity(entity)
        
        relations = [
            Relation("rel_1", "person_1", "company_1", "WORKS_AT", {"role": "Senior Engineer"}),
            Relation("rel_2", "company_1", "location_1", "LOCATED_IN", {}),
            Relation("rel_3", "person_1", "project_1", "WORKS_ON", {"contribution": "lead developer"}),
            Relation("rel_4", "person_1", "event_1", "ATTENDS", {"purpose": "presenting"})
        ]
        
        for relation in relations:
            graph_rag.add_relation(relation)
    
    # Show statistics and limits
    stats = graph_rag.get_statistics()
    limits = graph_rag.get_limits()
    
    print(f"\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nVisualization Limits:")
    for key, value in limits.items():
        print(f"  {key}: {value}")
    
    # Example: You can change limits dynamically
    # graph_rag.set_limits(max_nodes=300, max_edges=200)
    
    # Create visualization based on mode
    if mode.lower() == '2d':
        print(f"\n🎯 Creating 2D visualization...")
        print("   📊 Matplotlib-based 2D network graph")
        print("   🎨 Color-coded by entity type")
        
        graph_rag.visualize_2d_interactive("knowledge_graph_2d.png")
        
        print(f"\n✓ 2D visualization completed!")
        print(f"Files created:")
        print(f"- knowledge_graph_2d.png")
        
    else:  # Default to 3D
        print(f"\n🎯 Creating INTERACTIVE Mayavi 3D visualization...")
        print("   📌 CLICK ON ANY POINT to see entity details!")
        print("   🔄 Use mouse to rotate, zoom, and explore")
        print("   ❌ Close the window when done viewing")
        
        graph_rag.visualize_mayavi_3d_interactive("knowledge_graph_mayavi_interactive.png")
        
        print(f"\n✓ Interactive Mayavi visualization completed!")
        print(f"Files created:")
        print(f"- knowledge_graph_mayavi_interactive.png")
    
    return graph_rag

if __name__ == "__main__":
    import sys
    
    # Default values
    max_nodes = 500
    max_edges = 500
    mode = '3d'
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        try:
            max_nodes = int(sys.argv[1])
            print(f"Using custom max_nodes: {max_nodes}")
        except ValueError:
            # Check if first argument is mode
            if sys.argv[1].lower() in ['2d', '3d']:
                mode = sys.argv[1].lower()
                print(f"Using visualization mode: {mode}")
            else:
                print(f"Invalid argument: {sys.argv[1]}, using defaults")
    
    if len(sys.argv) > 2:
        try:
            max_edges = int(sys.argv[2])
            print(f"Using custom max_edges: {max_edges}")
        except ValueError:
            # Check if second argument is mode
            if sys.argv[2].lower() in ['2d', '3d']:
                mode = sys.argv[2].lower()
                print(f"Using visualization mode: {mode}")
            else:
                print(f"Invalid max_edges argument: {sys.argv[2]}, using default: {max_edges}")
    
    if len(sys.argv) > 3:
        if sys.argv[3].lower() in ['2d', '3d']:
            mode = sys.argv[3].lower()
            print(f"Using visualization mode: {mode}")
    
    # Print usage info
    if len(sys.argv) == 1:
        print(f"Usage: python {sys.argv[0]} [max_nodes] [max_edges] [mode]")
        print(f"       python {sys.argv[0]} [mode] (uses default limits)")
        print(f"Current defaults: max_nodes={max_nodes}, max_edges={max_edges}, mode={mode}")
        print(f"Mode options: 2d, 3d")
        print(f"Examples:")
        print(f"  python {sys.argv[0]} 300 200 2d  # Custom limits, 2D mode")
        print(f"  python {sys.argv[0]} 2d          # Default limits, 2D mode")
        print(f"  python {sys.argv[0]} 100 50      # Custom limits, 3D mode")
        print()
    
    demo_mayavi_knowledge_graph(max_nodes=max_nodes, max_edges=max_edges, mode=mode) 