from agent import build_graph


def run_research_pipeline(topic: str):
    app = build_graph()

    print("\n🚀 Running Research Pipeline...\n")

    result = app.invoke({
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": ""
    })

    print("\n================ REPORT ================\n")
    print(result["report"])

    print("\n================ CRITIC ================\n")
    print(result["feedback"])

    return result


if __name__ == "__main__":
    topic = input("Enter topic: ")
    run_research_pipeline(topic)