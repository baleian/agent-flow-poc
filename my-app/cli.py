import os
import json
import uuid
import time
from dotenv import load_dotenv
from typing import Any, Dict, Iterator, List

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown

# AIMessage, AIMessageChunk, ToolMessage 등을 직접 처리하기 위해 import
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, AIMessage, AIMessageChunk


class ConsoleUI:
    """LangGraph 스트리밍을 위한 다채로운 콘솔 UI 클래스"""

    def __init__(self, graph_app: Any):
        """
        초기화 메서드

        Args:
            graph_app: 컴파일된 LangGraph 애플리케이션 객체
        """
        self.app = graph_app
        self.console = Console()
        self.thread_id = None

    def _print_logo(self):
        """Cudori 로고를 출력합니다."""
        logo_text = Text("Cudori", style="bold magenta")
        panel = Panel(logo_text, title="[bold green]Agent[/bold green]", subtitle="[cyan]Welcome![/cyan]", border_style="green")
        self.console.print(panel)
        self.console.print("무엇을 도와드릴까요? (종료: 'exit' 또는 'quit', 새 대화: '/new')\n", style="italic yellow")

    @staticmethod
    def _truncate_text(text: str, start_len: int = 100, end_len: int = 100) -> str:
        """긴 텍스트의 앞뒤 일부만 남기고 중간을 ...으로 축약합니다."""
        if len(text) <= start_len + end_len + 5:
            return text
        return f"{text[:start_len]}...{text[-end_len:]}"

    def _handle_stream(self, stream: Iterator[Any]):
        """
        'messages' 스트림을 실시간으로 처리하여 자동 스크롤 효과를 구현합니다.
        (출력 순서 버그를 수정한 버전)
        """
        # --- 상태 변수 ---
        streaming_reasoning = ""
        streaming_response = ""
        current_tool_call = {"name": "", "args": ""}
        last_node_name = None
        is_reasoning_active = False
        is_response_active = False

        # Live 객체는 현재 스트리밍 중인 패널만 관리합니다.
        with Live(console=self.console, auto_refresh=False, transient=True) as live:
            for chunk, node_metadata in stream:
                # --- 노드 변경 시 이전 노드의 결과물을 먼저 출력 ---
                current_node = node_metadata.get('langgraph_node', '')
                if current_node and current_node != last_node_name:
                    # Live 스트리밍 중이던 내용을 먼저 완성하고 출력합니다.
                    if is_reasoning_active:
                        live.update("") # Live 내용을 지웁니다.
                        self.console.print(Panel(Markdown(streaming_reasoning), title="[cyan]Reasoning[/cyan]", border_style="cyan", expand=False))
                        streaming_reasoning = ""
                        is_reasoning_active = False

                    # Tool Call 정보가 있었다면 완성하고 출력합니다.
                    if current_tool_call.get("name"):
                        try:
                            args_str = current_tool_call.get('args', '{}')
                            parsed_args = json.loads(args_str)
                            pretty_args = json.dumps(parsed_args, indent=2, ensure_ascii=False)
                            tool_str = f"[bold]Tool:[/bold] {current_tool_call['name']}\n[bold]Args:[/bold]\n{pretty_args}"
                        except (json.JSONDecodeError, TypeError):
                            tool_str = f"[bold]Tool:[/bold] {current_tool_call['name']}\n[bold]Args:[/bold] {current_tool_call.get('args', '')}"
                        self.console.print(Panel(tool_str, title="[yellow]Tool Call[/yellow]", border_style="yellow", expand=False))
                        current_tool_call = {}

                    # 이제 새로운 노드의 상태 메시지를 출력합니다.
                    last_node_name = current_node
                    step = node_metadata.get('langgraph_step', '')
                    self.console.print(f"\n[grey50](Step {step}) Executing Node: {current_node}...[/grey50]")

                if isinstance(chunk, AIMessageChunk):
                    # 1. Reasoning 내용 스트리밍
                    if new_reasoning_chunk := chunk.additional_kwargs.get("reasoning_content"):
                        is_reasoning_active = True
                        streaming_reasoning += new_reasoning_chunk
                        live.update(Panel(Markdown(streaming_reasoning), title="[cyan]Reasoning[/cyan]", border_style="cyan", expand=False), refresh=True)

                    # 2. Tool Call 정보 조립
                    if chunk.tool_calls:
                        tool_chunk = chunk.tool_calls[0]
                        if 'name' in tool_chunk and tool_chunk['name']: current_tool_call['name'] = tool_chunk['name']
                        if 'args' in tool_chunk and tool_chunk['args']:
                            args_chunk = tool_chunk['args']
                            if isinstance(args_chunk, dict): current_tool_call['args'] = json.dumps(args_chunk, ensure_ascii=False)
                            else: current_tool_call['args'] = current_tool_call.get('args', '') + args_chunk
                    
                    # 3. 최종 답변 스트리밍
                    if chunk.content:
                        if is_reasoning_active:
                            live.update("") # Live 내용을 지웁니다.
                            self.console.print(Panel(Markdown(streaming_reasoning), title="[cyan]Reasoning[/cyan]", border_style="cyan", expand=False))
                            streaming_reasoning = ""
                            is_reasoning_active = False
                        
                        is_response_active = True
                        streaming_response += chunk.content
                        live.update(Panel(Markdown(streaming_response), title="[magenta]Cudori[/magenta]", border_style="magenta"), refresh=True)
                
                # 4. Tool 실행 결과 수신
                elif isinstance(chunk, ToolMessage):
                    # ToolMessage는 항상 새로운 노드('tools')에서 오므로,
                    # 노드 변경 로직에서 이미 이전 단계(Reasoning, Tool Call)가 처리되었습니다.
                    # 여기서는 Tool Result만 출력합니다.
                    tool_outputs = chunk.content
                    if not isinstance(tool_outputs, list): tool_outputs = [tool_outputs]
                    panel_content = ""
                    for output in tool_outputs:
                        if isinstance(output, dict):
                            metadata = output.get('metadata', {})
                            content = output.get('page_content', str(output))
                            truncated_content = self._truncate_text(content)
                            panel_content += f"[bold]Metadata:[/bold] {metadata}\n[bold]Content:[/bold]\n{truncated_content}\n---\n"
                        else:
                            panel_content += f"{str(output)}\n---\n"
                    self.console.print(Panel(panel_content.strip(), title="[green]Tool Result[/green]", border_style="green", expand=False))

        # 스트림 루프가 끝난 후, 마지막으로 스트리밍되던 내용을 콘솔에 출력합니다.
        if streaming_reasoning:
            self.console.print(Panel(Markdown(streaming_reasoning), title="[cyan]Reasoning[/cyan]", border_style="cyan", expand=False))
        if streaming_response:
            self.console.print(Panel(Markdown(streaming_response), title="[magenta]Cudori[/magenta]", border_style="magenta"))

    def run(self):
        """사용자 입력을 받고 에이전트를 실행하는 메인 루프"""
        self._print_logo()
        
        while True:
            try:
                user_input = self.console.input("[bold green]You: [/bold green]")

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
                
                stream = self.app.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages"
                )
                
                self._handle_stream(stream)
                
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
        from my_agent.agent import graph
        ui = ConsoleUI(graph)
        ui.run()
    except ImportError:
        print("오류: 'my_agent.agent'에서 'graph'를 찾을 수 없습니다.")
        print("agent.py 파일의 위치와 'graph' 객체의 이름을 확인해주세요.")
    except Exception as e:
        print(f"에이전트 실행 중 오류가 발생했습니다: {e}")
