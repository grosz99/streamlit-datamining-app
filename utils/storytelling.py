import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import json

class DashboardBuilder:
    def __init__(self):
        """Initialize the dashboard builder"""
        if 'dashboard_components' not in st.session_state:
            st.session_state.dashboard_components = []
        if 'component_order' not in st.session_state:
            st.session_state.component_order = []
    
    def add_visualization(self, fig, title, description=None, component_id=None):
        """
        Add a visualization to the dashboard
        
        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            The plotly figure to add
        title : str
            Title of the visualization
        description : str, optional
            Description or annotation for the visualization
        component_id : str, optional
            Unique identifier for the component
        """
        if component_id is None:
            component_id = f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        component = {
            'id': component_id,
            'type': 'visualization',
            'title': title,
            'description': description,
            'figure': fig.to_json()  # Convert to JSON for storage
        }
        
        st.session_state.dashboard_components.append(component)
        if component_id not in st.session_state.component_order:
            st.session_state.component_order.append(component_id)
    
    def add_text(self, content, title=None, component_id=None):
        """
        Add a text block to the dashboard
        
        Parameters:
        -----------
        content : str
            The text content to add
        title : str, optional
            Title for the text block
        component_id : str, optional
            Unique identifier for the component
        """
        if component_id is None:
            component_id = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        component = {
            'id': component_id,
            'type': 'text',
            'title': title,
            'content': content
        }
        
        st.session_state.dashboard_components.append(component)
        if component_id not in st.session_state.component_order:
            st.session_state.component_order.append(component_id)
    
    def remove_component(self, component_id):
        """Remove a component from the dashboard"""
        st.session_state.dashboard_components = [
            c for c in st.session_state.dashboard_components if c['id'] != component_id
        ]
        st.session_state.component_order.remove(component_id)
    
    def reorder_components(self, new_order):
        """Update the order of dashboard components"""
        st.session_state.component_order = new_order
    
    def get_component(self, component_id):
        """Get a specific component by ID"""
        for component in st.session_state.dashboard_components:
            if component['id'] == component_id:
                return component
        return None
    
    def clear_dashboard(self):
        """Clear all components from the dashboard"""
        st.session_state.dashboard_components = []
        st.session_state.component_order = []
    
    def render_dashboard(self):
        """Render all dashboard components in their current order"""
        for component_id in st.session_state.component_order:
            component = self.get_component(component_id)
            if component:
                with st.expander(f"{component['title']}", expanded=True):
                    if component['type'] == 'visualization':
                        fig = go.Figure(json.loads(component['figure']))
                        st.plotly_chart(fig, use_container_width=True)
                        if component.get('description'):
                            st.caption(component['description'])
                    elif component['type'] == 'text':
                        st.write(component['content'])
    
    def export_dashboard(self, format='html'):
        """
        Export the dashboard to the specified format
        Currently supports: 'html'
        
        Returns:
        --------
        str : The dashboard content in the specified format
        """
        if format == 'html':
            html_content = ['<html><head><title>Data Story Dashboard</title></head><body>']
            
            for component_id in st.session_state.component_order:
                component = self.get_component(component_id)
                if component:
                    html_content.append(f"<h2>{component['title']}</h2>")
                    
                    if component['type'] == 'visualization':
                        fig = go.Figure(json.loads(component['figure']))
                        html_content.append(fig.to_html(full_html=False))
                        if component.get('description'):
                            html_content.append(f"<p><em>{component['description']}</em></p>")
                    elif component['type'] == 'text':
                        html_content.append(f"<p>{component['content']}</p>")
            
            html_content.append('</body></html>')
            return '\n'.join(html_content)
        
        raise ValueError(f"Unsupported export format: {format}")

    def store_current_visualization(self, fig, auto_title=None):
        """
        Store the current visualization in session state
        
        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            The plotly figure to store
        auto_title : str, optional
            Automatic title to use if needed
        """
        if auto_title is None:
            auto_title = f"Analysis {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        st.session_state.last_figure = {
            'fig': fig,
            'auto_title': auto_title
        }
    
    def pin_current_visualization(self, title=None, description=None):
        """
        Pin the current visualization to the dashboard
        
        Parameters:
        -----------
        title : str, optional
            Title for the visualization (if None, uses auto_title)
        description : str, optional
            Description for the visualization
        
        Returns:
        --------
        bool : Whether the operation was successful
        """
        if 'last_figure' not in st.session_state:
            return False
            
        last_fig = st.session_state.last_figure
        viz_title = title if title else last_fig['auto_title']
        
        self.add_visualization(
            last_fig['fig'],
            viz_title,
            description
        )
        return True

def create_default_layout():
    """Create a default dashboard layout"""
    return {
        'title': 'Data Story Dashboard',
        'description': 'A collection of insights and visualizations',
        'layout': [
            {'width': 1, 'components': []},  # Full width
            {'width': 0.5, 'components': []},  # Half width
            {'width': 0.5, 'components': []}   # Half width
        ]
    }
