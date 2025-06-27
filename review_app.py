import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Union, List, Dict, Any
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# --- Configuration ---
MAPPED_DATA_PATH = Path(__file__).resolve().parent / "data" / "outputs" / "mapped_ingredients_output.json"

# --- Data Loading (Cached) ---
@st.cache_data
def load_data(file_path: Path) -> Union[Dict, None]:
    """Loads the mapped ingredients data from the specified JSON file."""
    if not file_path.exists():
        st.error(f"❌ **File Not Found:** The mapping file at '{file_path}' does not exist.")
        st.info("Please ensure you have run the appropriate scripts to generate the data.")
        return None
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"❌ **Error Loading Data:** Could not read or parse '{file_path}'. Reason: {e}")
        return None

# --- UI Helper Functions ---
def display_details_drawer(selected_row_data: Dict[str, Any]):
    """Renders the details panel for a selected ingredient mapping."""
    st.divider()
    st.header(f"Details for: `{selected_row_data['original_ingredient']}`")

    mapping = selected_row_data.get('mapping_result', {})
    if not isinstance(mapping, dict):
        st.warning("No valid mapping data found for this entry.")
        return

    st.subheader(f"✅ Chosen Term: {mapping.get('label', 'N/A')}")
    st.markdown(f"**CURIE:** `{mapping.get('id', 'N/A')}` 📑")
    
    if mapping.get('definition'):
        st.markdown(f"**ℹ️ Definition:** {mapping.get('definition')}")
    
    if mapping.get('synonyms'):
        st.markdown(f"**Synonyms:** *{', '.join(mapping.get('synonyms'))}*")
    
    st.divider()

    st.subheader("Model Explanation")
    st.info(mapping.get('explanation', 'No explanation provided.'))

    st.subheader("Candidates Considered")
    candidates = selected_row_data.get('candidates', [])
    if candidates:
        sort_key = 'rerank_score' if any('rerank_score' in c for c in candidates) else 'score'
        
        cand_df = pd.DataFrame(candidates)
        cand_df['Is Chosen'] = cand_df['id'].apply(lambda x: '⭐' if x == mapping.get('id') else '')
        
        display_cols = {'Is Chosen': 'Chosen', 'label': 'Label', 'id': 'CURIE', sort_key: 'Score', 'source_ontology': 'Source'}
        existing_cols = [col for col in display_cols.keys() if col in cand_df.columns]
        cand_df_display = cand_df[existing_cols].rename(columns=display_cols)

        st.dataframe(cand_df_display, use_container_width=True, hide_index=True, column_config={"Score": st.column_config.NumberColumn(format="%.3f")})
    else:
        st.info("No other candidates were provided for this mapping.")

    with st.expander("Show Raw JSON Data"):
        st.json(selected_row_data)

# JsCode for cell styling in AgGrid
cellsytle_jscode = JsCode("""
function(params) {
    function interpolateColor(color1, color2, factor) {
        let result = color1.slice();
        for (let i = 0; i < 3; i++) {
            result[i] = Math.round(result[i] + factor * (color2[i] - color1[i]));
        }
        return result;
    }

    let confidence = params.data.Confidence;
    if (confidence === null || confidence === undefined) {
        return { backgroundColor: '#f0f0f0' };
    }
    
    const red_pastel = [255, 224, 224];
    const green_pastel = [224, 255, 224];
    
    let color = interpolateColor(red_pastel, green_pastel, confidence);
    
    return {
        'backgroundColor': `rgb(${color[0]}, ${color[1]}, ${color[2]})`
    };
};
""")

# --- DataFrame Preparation ---
# *** CHANGE: Made this function robust against non-dict mapping_result values ***
def prepare_dataframe(mapped_ingredients: List[Dict]) -> pd.DataFrame:
    """Transforms the list of mapping results into a DataFrame for AgGrid."""
    records = []
    for i, item in enumerate(mapped_ingredients):
        item['_id'] = i
        mapping = item.get('mapping_result')

        # Check if the mapping is a valid dictionary with an ID.
        # This prevents errors if mapping_result is a string (e.g., "UNMAPPED") or None.
        if isinstance(mapping, dict) and mapping.get('id'):
            term = mapping.get('label', '⚠️ UNMAPPED')
            ont_id = mapping.get('id', 'N/A')
            confidence = mapping.get('confidence_score', 0.0)
            explanation = mapping.get('explanation', 'No explanation provided.')
        else:
            # If not a valid mapping, set all values to their "unmapped" defaults.
            term = '⚠️ UNMAPPED'
            ont_id = 'N/A'
            confidence = 0.0
            explanation = 'No valid mapping result found for this ingredient.'

        record = {
            "Token": item['original_ingredient'],
            "Ontology Term": term,
            "Ontology ID": ont_id,
            "Confidence": confidence,
            "Explanation": explanation,
            "_id": i
        }
        records.append(record)
    
    return pd.DataFrame(records)

# --- Main Application ---
st.set_page_config(page_title="Mapping Reviewer", page_icon="🔎", layout="wide")

st.title("🔎 Ontology Mapping Review")
st.markdown("An interactive interface to review, validate, and correct model-generated ontology mappings.")

data = load_data(MAPPED_DATA_PATH)

if data:
    if 'product_keys' not in st.session_state:
        st.session_state.product_keys = list(data.keys())
        st.session_state.current_index = 0
        
    with st.sidebar:
        st.header("Product Navigation")
        selected_product_key = st.selectbox(
            "Select Product to Review:",
            options=st.session_state.product_keys,
            index=st.session_state.current_index,
            key="product_selector"
        )
        new_index = st.session_state.product_keys.index(selected_product_key)
        if new_index != st.session_state.current_index:
            st.session_state.current_index = new_index
            st.rerun()

        col1, col2 = st.columns(2)
        if col1.button("⬅️ Previous", use_container_width=True, disabled=(st.session_state.current_index == 0)):
            st.session_state.current_index -= 1
            st.rerun()
        if col2.button("Next ➡️", use_container_width=True, disabled=(st.session_state.current_index >= len(st.session_state.product_keys) - 1)):
            st.session_state.current_index += 1
            st.rerun()
        
        st.divider()
        st.caption(f"Progress: Product {st.session_state.current_index + 1} of {len(st.session_state.product_keys)}")
        st.progress((st.session_state.current_index + 1) / len(st.session_state.product_keys))

    product_id = st.session_state.product_keys[st.session_state.current_index]
    product_data = data[product_id]
    mapped_ingredients_list = product_data.get('mapped_ingredients', [])

    st.header(f"Reviewing Product: `{product_id}`", divider="rainbow")
    
    if mapped_ingredients_list:
        df_for_metrics = prepare_dataframe(mapped_ingredients_list)
        total_ingredients = len(df_for_metrics)
        mapped_count = len(df_for_metrics[df_for_metrics['Confidence'] > 0])
        avg_confidence = df_for_metrics['Confidence'][df_for_metrics['Confidence'] > 0].mean()
        low_confidence_count = len(df_for_metrics[df_for_metrics['Confidence'].between(0.01, 0.5, inclusive="right")])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Tokens", f"{total_ingredients}")
        c2.metric("Mapped", f"{mapped_count}/{total_ingredients}")
        c3.metric("Avg. Confidence", f"{avg_confidence:.2f}" if pd.notna(avg_confidence) else "N/A")
        c4.metric("Low Confidence (<0.5)", f"{low_confidence_count}")
        
    full_ingredients = product_data.get("original_ingredients", "N/A")
    
    st.markdown("**Original Ingredients String:**")
    st.code(full_ingredients, language=None)
    st.markdown("---")
    
    st.subheader("Ingredient Mappings")
    st.info("🎨 Row color indicates confidence.  Hover over 'Ontology Term' or 'ID' for the model's explanation. Click a row to see full details below.")
    
    if not mapped_ingredients_list:
        st.warning("No mapped ingredients found for this product.")
    else:
        df_for_display = prepare_dataframe(mapped_ingredients_list)

        gb = GridOptionsBuilder.from_dataframe(df_for_display)
        
        gb.configure_column("Ontology Term", tooltipField="Explanation")
        gb.configure_column("Ontology ID", tooltipField="Explanation")
        gb.configure_default_column(cellStyle=cellsytle_jscode)
        
        gb.configure_selection(selection_mode="single", use_checkbox=False)
        gb.configure_grid_options(domLayout='normal')
        
        gb.configure_column("_id", hide=True)
        gb.configure_column("Confidence", hide=True)
        gb.configure_column("Explanation", hide=True)
        
        grid_options = gb.build()

        grid_response = AgGrid(
            df_for_display,
            gridOptions=grid_options,
            height=400,
            width='100%',
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            key=f"aggrid_{product_id}"
        )

        selected_row_data = None
        
        if grid_response['selected_rows'] is not None and not grid_response['selected_rows'].empty:
            selected_id = grid_response['selected_rows'].iloc[0]['_id']
            selected_row_data = next((item for item in mapped_ingredients_list if item.get('_id') == selected_id), None)
        elif mapped_ingredients_list:
            selected_row_data = mapped_ingredients_list[0]

        if selected_row_data:
            display_details_drawer(selected_row_data)

else:
    st.warning("Could not load data. Please check the file path and format.")