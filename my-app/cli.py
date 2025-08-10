import os
import json
import uuid
import asyncio
from dotenv import load_dotenv
from typing import Any, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown

from langchain_core.messages import HumanMessage


class ConsoleUI:

    def __init__(self, graph_app: Any):
        self.app = graph_app
        self.console = Console()
        self.thread_id = None

    def _print_logo(self):
        logo_text = Text("Cudori", style="bold magenta")
        panel = Panel(logo_text, title="[bold green]Agent[/bold green]", subtitle="[cyan]Welcome![/cyan]", border_style="green")
        self.console.print(panel)
        self.console.print("무엇을 도와드릴까요? (종료: 'exit' 또는 'quit', 새 대화: '/new')\n", style="italic yellow")

    @staticmethod
    def _truncate_text(text: str, start_len: int = 100, end_len: int = 100) -> str:
        if len(text) <= start_len + end_len + 5:
            return text
        return f"{text[:start_len]}...{text[-end_len:]}"

    async def _handle_stream(self, stream: Iterator[Any]):
        streaming_reasoning = ""
        streaming_content = ""

        # Live 객체는 현재 스트리밍 중인 패널만 관리합니다.
        with Live(console=self.console, auto_refresh=False, transient=True) as live:
            async for event in stream:
                kind = event["event"]
                node_metadata = event["metadata"]
                
                if kind == "on_chat_model_start":
                    self.console.print(f"[grey50] Executing Node: {node_metadata['langgraph_node']}...[/grey50]")

                elif kind == "on_chat_model_stream":
                    data = event["data"]
                    chunk = data["chunk"]
                    if new_reasoning_chunk := chunk.additional_kwargs.get("reasoning_content"):
                        streaming_reasoning += new_reasoning_chunk
                        live.update(Panel(streaming_reasoning, title="[cyan]Reasoning[/cyan]", border_style="cyan"), refresh=True)
                    elif chunk.content:
                        if streaming_reasoning:
                            self.console.print(Panel(streaming_reasoning, title="[cyan]Reasoning[/cyan]", border_style="cyan"))
                            live.update(Panel("", title="[cyan]Reasoning[/cyan]", border_style="cyan"), refresh=True)
                            streaming_reasoning = ""
                        streaming_content += chunk.content
                        live.update(Panel(Markdown(streaming_content), title="[magenta]Cudori[/magenta]", border_style="magenta"), refresh=True)
                
                elif kind == "on_chat_model_end":
                    if streaming_reasoning:
                        self.console.print(Panel(streaming_reasoning, title="[cyan]Reasoning[/cyan]", border_style="cyan"))
                        live.update(Panel("", title="[cyan]Reasoning[/cyan]", border_style="cyan"), refresh=True)
                        streaming_reasoning = ""
                    if streaming_content:
                        self.console.print(Panel(Markdown(streaming_content), title="[magenta]Cudori[/magenta]", border_style="magenta"))
                        live.update(Panel("", title="[magenta]Cudori[/magenta]", border_style="magenta"))
                        streaming_content = ""

                elif kind == "on_tool_start":
                    self.console.print(f"[grey50] Tool Calling: {node_metadata['langgraph_node']} ({event['name']})...[/grey50]")
                       
                elif kind == "on_tool_end":
                    data = event["data"]
                    if "input" in data:
                        tool_msg = data["input"]
                        pretty_args = json.dumps(tool_msg, indent=2, ensure_ascii=False)
                        tool_str = f"[bold]Tool:[/bold] {event['name']}\n[bold]Args:[/bold]\n{pretty_args}"
                        self.console.print(Panel(tool_str, title="[yellow]Tool Call[/yellow]", border_style="yellow", expand=False))
                    if "output" in data:
                        self.console.print(Panel(data["output"].content, title="[green]Tool Result[/green]", border_style="green", expand=False))

    def run(self):
        """사용자 입력을 받고 에이전트를 실행하는 메인 루프"""
        self._print_logo()
        
        while True:
            try:
                user_input = self.console.input("[bold green]You: [/bold green]")

                if not user_input.rstrip():
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    self.console.print("[bold red]Cudori를 종료합니다.[/bold red]")
                    break

                if user_input.lower() == "/new":
                    self.thread_id = None
                    self.console.print("[yellow]새로운 대화를 시작합니다.[/yellow]")
                    self.console.print("-" * 50, style="dim")
                    continue
                
                # 새 대화 시작 시 thread_id 생성
                if self.thread_id is None:
                    self.thread_id = uuid.uuid4()
                    self.console.print(f"[yellow]New conversation started. Thread ID: {self.thread_id}[/yellow]")

                self.console.print("-" * 50, style="dim")

                # MemorySaver를 위한 config 객체 생성
                config = {
                    "configurable": {
                        "thread_id": str(self.thread_id),
                    }
                }
                
                stream = self.app.astream_events(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    subgraphs=True
                )
                
                asyncio.run(self._handle_stream(stream))
                
                self.console.print("\n" + "-" * 50, style="dim")

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Cudori를 종료합니다.[/bold red]")
                break
            except Exception as e:
                self.console.print(f"[bold red]오류가 발생했습니다:[/bold red] {e}")


if __name__ == "__main__":
    if os.path.exists(".env"):
        load_dotenv()

    try:
        from app.chatbot import make_chatbot_graph
        graph = make_chatbot_graph()
        with open("graph.png", "wb") as f:
            f.write(graph.get_graph(xray=True).draw_mermaid_png())
        ui = ConsoleUI(graph)
        ui.run()
    except ImportError as e:
        print(e)
        print("오류: 'graph'를 찾을 수 없습니다.")
    except Exception as e:
        print(f"에이전트 실행 중 오류가 발생했습니다: {e}")


# async def main():
#     from app.agents.supervisor import graph
#     inputs = {"messages": [HumanMessage(content="버블소트 알고리즘 파이썬으로 작성해줘")]}
#     config = {"configurable": {"thread_id": "test123"}}
#     event_streamer = graph.astream_events(inputs, config=config, subgraphs=True)
#     async for event in event_streamer:
#         kind = event['event']

#         if kind == "on_chat_model_start":
#             print(f"\n========= on_chat_model_start =========\n")
#             print(event['metadata']['langgraph_node'])
#             print(f"\n=======================================\n")

#         # 채팅 모델 스트림 이벤트 및 최종 노드 태그 필터링
#         elif kind == "on_chat_model_stream":
#             # 이벤트 데이터 추출
#             data = event["data"]

#             # 토큰 단위의 스트리밍 출력
#             if data["chunk"].content:
#                 print(data["chunk"].content, end="", flush=True)

#         elif kind == "on_tool_start":
#             print(f"\n========= tool_start =========\n")
#             print(event['name'])
#             print(event['metadata']['langgraph_node'])
#             print(f"\n==============================\n")
#             data = event["data"]
#             if "input" in data:
#                 tool_msg = data["input"]
#                 print(tool_msg)            

#         elif kind == "on_tool_end":
#             print(f"\n========= tool_end =========\n")
#             print(event['name'])
#             print(event['metadata']['langgraph_node'])
#             print(f"\n============================\n")
#             data = event["data"]
#             if "output" in data:
#                 tool_msg = data["output"]
#                 print(tool_msg.content)


# if __name__ == "__main__":
#     import asyncio
#     load_dotenv()
#     asyncio.run(main())
