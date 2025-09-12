import asyncio
import streamlit as st

from src.adapters.pipeline_factory import create_pipeline
from src.application.pipeline_verbose import run_pipeline_verbose

st.set_page_config(page_title="Onto-RAG Live Demo", page_icon="🧠")
st.title("🔍 Onto-RAG Interactive Pipeline")

provider = st.selectbox("LLM Provider", ["gemini", "ollama", "huggingface"], index=0)
query = st.text_input("Entity to map")
run = st.button("Run Pipeline")

if run:
    if not query:
        st.warning("Please enter an entity to map.")
    else:
        pipeline = create_pipeline(provider)
        with st.spinner("Running pipeline..."):
            final_result, candidates, history = asyncio.run(
                run_pipeline_verbose(pipeline, query)
            )
        pipeline.close()

        for idx, step in enumerate(history, 1):
            st.header(f"Iteration {idx}: '{step['query']}'")
            st.subheader("Retrieved Entities")
            st.json(step.get("retrieved_entities", []))

            st.subheader("Selector Prompt")
            st.code(step.get("selector_prompt", ""))
            st.subheader("Selector Response")
            st.code(step.get("selector_raw_response", ""))
            st.subheader("Selector Result")
            st.json(step.get("selection", {}))

            if step.get("scorer_prompt"):
                st.subheader("Scorer Prompt")
                st.code(step.get("scorer_prompt", ""))
                st.subheader("Scorer Response")
                st.code(step.get("scorer_raw_response", ""))
                st.subheader("Scorer Result")
                st.json(step.get("scorer_result", {}))
                if step.get("scorer_suggestions"):
                    st.subheader("Scorer Suggestions")
                    st.write(step["scorer_suggestions"])

            if step.get("synonyms"):
                st.subheader("Synonym Prompt")
                st.code(step.get("synonym_prompt", ""))
                st.subheader("Synonym Response")
                st.code(step.get("synonym_raw_response", ""))
                st.subheader("Generated Synonyms")
                st.write(step["synonyms"])

        st.header("Final Result")
        st.json(final_result)
