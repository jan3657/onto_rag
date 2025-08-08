import streamlit as st
import json
from pathlib import Path

# --- Configuration ---
# Set the path to the human-readable evaluation results file.
# This script assumes the file is in the 'data' subdirectory.
EVALUATION_FILE_PATH = Path("data") / "readable_evaluation_results_gemini.json"

# --- Helper Functions ---

@st.cache_data
def load_data(file_path: Path) -> list:
    """
    Loads the evaluation data from the specified JSON file.
    The @st.cache_data decorator ensures the data is loaded only once.
    """
    if not file_path.exists():
        st.error(f"Error: Evaluation file not found at '{file_path}'.")
        st.info("Please run the `scripts/format_evaluation_results.py` script first to generate this file.")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def display_term_details(term_data: dict):
    """Renders the details of a single ontology term without a main title."""
    if not term_data or not term_data.get("curie"):
        st.warning("No data available for this term.")
        return

    # Display the label and CURIE
    label = term_data.get('label', 'N/A')
    curie = term_data.get('curie', 'N/A')
    st.markdown(f"**{label}** (`{curie}`)")

    # Display the definition in an info box
    definition = term_data.get('definition')
    if definition:
        st.info(f"**Definition:** {definition}")
    else:
        st.info("No definition provided.")

    # Display synonyms if they exist
    synonyms = term_data.get('synonyms', [])
    if synonyms:
        st.markdown(f"**Synonyms:** *{', '.join(synonyms)}*")

def display_term(term_data: dict, title: str):
    """Renders a single ontology term's details in a structured format."""
    st.subheader(title)
    display_term_details(term_data)

# --- Main Application Logic ---

# Set the page configuration (title, icon, layout)
st.set_page_config(
    page_title="Ontology Linking Review",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Ontology Linking Evaluation Review")
st.markdown("An interface for experts to review the performance of the entity linking model.")

# Load the data using the cached function
data = load_data(EVALUATION_FILE_PATH)

if data:
    # --- Session State for Navigation ---
    # Initialize the session state to keep track of the current item index
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # --- Navigation Controls ---
    st.sidebar.header("Navigation")
    # Allow selection by query text for easy lookup
    query_list = [f"{i+1}. {item['query']}" for i, item in enumerate(data)]
    selected_query = st.sidebar.selectbox("Select a Query to Review:", query_list, index=st.session_state.current_index)

    # Update index based on selection
    st.session_state.current_index = query_list.index(selected_query)

    col1, col2 = st.sidebar.columns(2)
    # "Previous" button
    if col1.button("⬅️ Previous", use_container_width=True):
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()
    # "Next" button
    if col2.button("Next ➡️", use_container_width=True):
        if st.session_state.current_index < len(data) - 1:
            st.session_state.current_index += 1
            st.rerun()

    # --- Display the selected item ---
    item = data[st.session_state.current_index]
    query = item.get("query")

    st.header(f"Reviewing Query: \"{query}\"", divider="rainbow")

    # --- CHANGE 1: Removed the Correct/Incorrect status message ---
    # The block checking item.get("is_correct") was removed from here.

    # Use columns for a side-by-side comparison
    left_col, right_col = st.columns(2)

    with left_col:
        # Display the model's chosen term
        display_term(item.get("chosen_term"), "🤖 Model's Choice")
        # Display the model's reasoning in an expandable section
        with st.expander("Show Model's Explanation"):
            st.info(item.get("explanation", "No explanation provided."))

    with right_col:
        # --- CHANGE 2: Display ALL ground truth terms with full details ---
        st.subheader("🎯 Ground Truth(s)")
        ground_truth_terms = item.get("ground_truth_terms", [])
        
        if not ground_truth_terms:
             st.warning("No ground truth terms provided for this query.")
        else:
            for i, term in enumerate(ground_truth_terms):
                # Add a separator between terms for clarity, but not before the first one
                if i > 0:
                    st.markdown("---")
                display_term_details(term)


    # --- Display the list of all candidates provided to the model ---
    st.markdown("---")
    with st.expander("🕵️‍♀️ View All Candidate Terms Provided to Model"):
        st.markdown("This is the full list of options the model had to choose from.")

        candidates = item.get("candidate_terms_provided", [])
        chosen_curie = item.get("chosen_term", {}).get("curie")
        ground_truth_curies = [gt.get("curie") for gt in item.get("ground_truth_terms", [])]

        if not candidates:
            st.info("No candidate terms were provided to the model for this query.")
        else:
            for candidate in candidates:
                label = candidate.get('label', 'N/A')
                curie = candidate.get('curie', 'N/A')

                # Highlight chosen and ground truth terms
                marker = ""
                if curie == chosen_curie:
                    marker += "🤖"
                if curie in ground_truth_curies:
                    marker += "🎯"

                st.markdown(f"**{marker} {label}** (`{curie}`)")
                definition = candidate.get('definition')
                if definition:
                    st.text(f"  - {definition[:200]}...") # Truncate long definitions
                else:
                    st.text("  - No definition.")