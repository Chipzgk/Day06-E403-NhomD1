from src.agent.graph import build_graph

INITIAL_STATE = {
    "concept": "",
    "interest": "",
    "clarified_interest": "",
    "confidence": 0.0,
    "narrative": "",
    "mapping_table": [],
    "is_fallback": False,
    "fallback_reason": "",
    "suggestions": [],
    "needs_clarification": False,
    "clarification_question": "",
}

def run(concept: str, interest: str):
    graph = build_graph()
    result = graph.invoke({**INITIAL_STATE, "concept": concept, "interest": interest})

    # Cần hỏi lại sở thích
    if result["needs_clarification"]:
        print(f"\n❓ {result['clarification_question']}")
        new_interest = input("   Nhập sở thích cụ thể hơn: ").strip()
        result = graph.invoke({**INITIAL_STATE, "concept": concept, "interest": new_interest})

    # Fallback
    if result["is_fallback"]:
        print(f"\n⚠️  Confidence thấp ({result['confidence']:.0%})")
        print(f"   Lý do: {result['fallback_reason']}")
        print("\n💡 Gợi ý sửa:")
        for i, s in enumerate(result["suggestions"], 1):
            print(f"   {i}. {s}")
        return

    # Kết quả thành công
    print(f"\n{'═'*55}")
    print(f"  ANALOGY: {result['concept']}  ←→  {result['interest']}")
    print(f"{'═'*55}")
    print(f"\n📖 Câu chuyện:\n{result['narrative']}")
    print(f"\n📊 Bảng Mapping:")
    for item in result["mapping_table"]:
        print(f"   [{item['technical_component']}]  =  [{item['analogy_element']}]")
    print(f"\n✅ Confidence: {result['confidence']:.0%}")


if __name__ == "__main__":
    concept  = input("Nhập khái niệm AI cần hiểu: ").strip()
    interest = input("Nhập sở thích cá nhân của bạn: ").strip()
    run(concept, interest)