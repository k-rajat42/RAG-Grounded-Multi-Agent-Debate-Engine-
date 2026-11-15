from src.graph.debate_graph import build_debate_graph

app = build_debate_graph()

png_bytes = app.get_graph().draw_mermaid_png()
with open("debate_graph.png", "wb") as f:
    f.write(png_bytes)

print("Saved debate_graph.png")