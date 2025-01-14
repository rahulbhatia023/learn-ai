from typing import TypedDict, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from openai import OpenAI
from pytubefix import YouTube
import os
import streamlit as st


class YoutubeVideoSummarizerState(TypedDict):
    video_url: str
    audio_file: str
    transcript: str
    summary: str


class YoutubeVideoSummarizerAgent:
    agent_name = "Youtube Video Summarizer"

    required_api_keys = {"OPENAI_API_KEY": "password"}

    @classmethod
    def get_graph(cls):
        client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])

        llm = ChatOpenAI(model_name="gpt-4o")

        def _default_po_token_verifier() -> Tuple[str, str]:
            visitor_data = str(st.secrets["YOUTUBE_VISITOR_DATA"])
            po_token = str(st.secrets["YOUTUBE_PO_TOKEN"])
            return visitor_data, po_token

        def download_audio(state: YoutubeVideoSummarizerState):
            video = YouTube(
                url=state["video_url"],
                token_file="/tmp",
                use_po_token=True,
                po_token_verifier=_default_po_token_verifier,
            )
            audio_file_name = video.streams.filter(only_audio=True).first().download()
            audio_file_base_name = os.path.basename(audio_file_name)
            name, extension = os.path.splitext(audio_file_base_name)
            audio_file = f"{name}.mp3"
            os.rename(audio_file_base_name, audio_file)
            return {"audio_file": audio_file}

        def transcribe_audio(state: YoutubeVideoSummarizerState):
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=open(state["audio_file"], "rb")
            ).text
            return {"transcript": transcript}

        def summarize_video(state: YoutubeVideoSummarizerState):
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                            You are an AI expert in summarizing video content based on provided transcript. 
                            Given a video transcript in plain text, your task is to generate a concise and engaging summary that highlights the key points, main topics, and any important insights discussed in the video. 
                            Structure the summary in a user-friendly format, such as a short paragraph, bullet points, or both, depending on the user’s preference. 
                            Focus on delivering the most relevant and actionable information from the transcript while omitting unnecessary details.
                        """,
                    ),
                    (
                        "human",
                        """
                            Here is the transcript of the video:
                            {transcript}
                            
                            Please summarize the video.
                        """,
                    ),
                ]
            )

            response = llm.invoke(prompt.format(transcript=state["transcript"])).content

            return {"summary": response}

        graph = StateGraph(state_schema=YoutubeVideoSummarizerState)

        graph.add_node("Download Audio", download_audio)
        graph.add_node("Transcribe Audio", transcribe_audio)
        graph.add_node("Summarize Video", summarize_video)

        graph.add_edge(START, "Download Audio")
        graph.add_edge("Download Audio", "Transcribe Audio")
        graph.add_edge("Transcribe Audio", "Summarize Video")
        graph.add_edge("Summarize Video", END)

        return graph.compile(checkpointer=MemorySaver())
