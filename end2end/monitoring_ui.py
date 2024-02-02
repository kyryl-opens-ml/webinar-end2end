import argilla as rg
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict
from text_generation import Client


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    ARGILLA_URI: str
    ARGILLA_KEY: str
    ARGILLA_NAMESPACE: str
    SERVING_URL: str
    FEEDBACK_DATASET_NAME: str


ARGILLA_URI = Settings().ARGILLA_URI
ARGILLA_KEY = Settings().ARGILLA_KEY
ARGILLA_NAMESPACE = Settings().ARGILLA_NAMESPACE
SERVING_URL = Settings().SERVING_URL
FEEDBACK_DATASET_NAME = Settings().FEEDBACK_DATASET_NAME


class FeedbackLoopClient:
    def __init__(self) -> None:
        rg.init(api_url=ARGILLA_URI, api_key=ARGILLA_KEY, workspace=ARGILLA_NAMESPACE)

        name = FEEDBACK_DATASET_NAME
        dataset = rg.FeedbackDataset(
            guidelines="Text to SurrealQL.",
            fields=[
                rg.TextField(name="user_query", title="Text from user"),
                rg.TextField(name="sql_schema", title="Schema", required=False),
                rg.TextField(name="sql_query", title="LLM output", required=False),
                rg.TextField(name="llm_text_input", title="Full LLM input", required=False),
                rg.TextField(name="serving_url", title="serving_url", required=False),
            ],
            questions=[
                rg.TextQuestion(
                    name="corrected_sq_query",
                    title="Provide a correction to the SurrealQL query:",
                    required=True,
                    use_markdown=True,
                ),
                rg.LabelQuestion(
                    name="is_ql_query_correct",
                    title="Is QL query correct?",
                    required=True,
                    labels=["correct", "not correct"],
                ),
            ],
        )
        if name in [x.name for x in rg.list_datasets()]:
            feedback_dataset = rg.FeedbackDataset.from_argilla(name=name)
        else:
            feedback_dataset = dataset.push_to_argilla(name=name)
        self.feedback_dataset = feedback_dataset

    def submit_feedback(
        self, table_schema: str, user_query: str, llm_text_input: str, generated_text: str, serving_url: str
    ):
        record = rg.FeedbackRecord(
            fields={
                "user_query": user_query,
                "sql_schema": table_schema,
                "sql_query": generated_text,
                "llm_text_input": llm_text_input,
                "serving_url": serving_url,
            }
        )
        self.feedback_dataset.add_records([record])

    def generate(self, schema: str, user_query: str) -> str:

        client = Client(SERVING_URL)
        prompt = "### SQL schema: {sql_schema} ### User query: {user_query} ### SQL query:"
        llm_text_input = prompt.format(sql_schema=schema, user_query=user_query)
        generated_text = client.generate(llm_text_input, max_new_tokens=64).generated_text

        self.submit_feedback(
            table_schema=schema,
            user_query=user_query,
            llm_text_input=llm_text_input,
            generated_text=generated_text,
            serving_url=SERVING_URL,
        )

        return generated_text


@st.cache_resource
def get_client():
    return FeedbackLoopClient()


def ui():
    client = get_client()
    schema = st.text_input("schema")
    user_query = st.text_input("user_query")

    st.write("The current schema is", schema)
    st.write("The current user_query", user_query)

    generate_ql = st.button("Generate SQL")
    if generate_ql:
        generated_ql = client.generate(schema=schema, user_query=user_query)
        st.write("Result:", generated_ql)


if __name__ == "__main__":
    ui()
