import streamlit as st
import base64
import os

class ModernUITheme:
    """
    A class for applying a modern UI theme to the Streamlit app.
    """

    def __init__(self):
        """Initialize the UI theme with default settings."""
        # Color scheme
        self.colors = {
            'primary': '#4F8BF9',
            'secondary': '#FF4B4B',
            'background': '#0E1117',
            'surface': '#1E2130',
            'text': '#FFFFFF',
            'text_secondary': '#B0B0B0',
            'success': '#00C851',
            'warning': '#FFBB33',
            'error': '#FF4444',
            'info': '#33B5E5'
        }

        # Font settings
        self.fonts = {
            'heading': '"Roboto", sans-serif',
            'body': '"Open Sans", sans-serif',
            'code': '"Fira Code", monospace'
        }

        # Spacing
        self.spacing = {
            'xs': '0.25rem',
            'sm': '0.5rem',
            'md': '1rem',
            'lg': '1.5rem',
            'xl': '2rem'
        }

    def apply_theme(self):
        """Apply the modern UI theme to the Streamlit app."""
        # Apply custom CSS
        st.markdown(self._get_css(), unsafe_allow_html=True)

    def add_logo(self, logo_path=None):
        """Add a logo to the sidebar."""
        if logo_path is None or not os.path.exists(logo_path):
            # Use a default logo
            logo_html = """
            <div style="display: flex; align-items: center; margin-bottom: 1rem; padding: 1rem;">
                <div style="font-size: 2rem; margin-right: 0.5rem;">🏥</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #4F8BF9;">Smart Hospital</div>
            </div>
            """
        else:
            # Use the provided logo
            logo_base64 = self._get_base64_encoded_image(logo_path)
            logo_html = f"""
            <div style="margin-bottom: 1rem; padding: 1rem;">
                <img src="data:image/png;base64,{logo_base64}" style="max-width: 100%; height: auto;">
            </div>
            """

        st.sidebar.markdown(logo_html, unsafe_allow_html=True)

    def add_footer(self):
        """Add a footer to the app."""
        footer_html = """
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1); text-align: center; font-size: 0.8rem; color: #B0B0B0;">
            <p>© 2025 Smart Hospital Analytics | Powered by Streamlit</p>
        </div>
        """
        st.markdown(footer_html, unsafe_allow_html=True)

    def create_card(self, title, content, icon=None):
        """Create a styled card with title and content."""
        # Clean up the content to remove any extra whitespace
        content = content.strip()

        if icon:
            card_html = f"""<div style="background-color: #1E2130; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1rem; overflow: hidden;"><div style="display: flex; align-items: center; padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.1);"><div style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</div><h3 style="margin: 0; color: #4F8BF9;">{title}</h3></div><div style="padding: 1rem;">{content}</div></div>"""
        else:
            card_html = f"""<div style="background-color: #1E2130; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1rem; overflow: hidden;"><div style="padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.1);"><h3 style="margin: 0; color: #4F8BF9;">{title}</h3></div><div style="padding: 1rem;">{content}</div></div>"""

        return card_html

    def create_info_box(self, message, type='info'):
        """Create a styled info box with a message."""
        icon_map = {
            'info': '📌',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }

        color_map = {
            'info': '#33B5E5',
            'success': '#00C851',
            'warning': '#FFBB33',
            'error': '#FF4444'
        }

        icon = icon_map.get(type, icon_map['info'])
        color = color_map.get(type, color_map['info'])

        # Clean up the message to remove any extra whitespace
        message = message.strip()

        info_box_html = f"""<div style="display: flex; align-items: flex-start; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; background-color: rgba(51, 181, 229, 0.1); border-left: 4px solid {color};"><div style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</div><div style="flex: 1;">{message}</div></div>"""

        return info_box_html

    def create_stat_card(self, title, value, delta=None, prefix='', suffix=''):
        """Create a styled stat card with title, value, and optional delta using Streamlit's native metric component."""
        # This function now returns nothing as we'll use Streamlit's native metric component directly
        # in the main app code
        pass

    def _get_css(self):
        """Get the custom CSS for the theme."""
        return """
        <style>
            /* Global styles */
            body {
                color: #FFFFFF;
                background-color: #0E1117;
            }

            h1, h2, h3, h4, h5, h6 {
                font-weight: 600;
            }

            /* Make the app wider */
            .reportview-container .main .block-container {
                max-width: 1200px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            /* Improve sidebar appearance */
            .sidebar .sidebar-content {
                background-color: #1E2130;
            }

            /* Improve button appearance */
            .stButton>button {
                border-radius: 0.5rem;
                font-weight: 600;
            }

            /* Improve selectbox appearance */
            .stSelectbox>div>div {
                border-radius: 0.5rem;
            }

            /* Improve slider appearance */
            .stSlider>div>div {
                border-radius: 0.5rem;
            }
        </style>
        """

    def _get_base64_encoded_image(self, image_path):
        """Get base64 encoded image from file path."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
