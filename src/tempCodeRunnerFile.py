
    print("Polished answer:", polished)

    # Test generate_polished_answer — question that CANNOT be answered from chunks
    unrelated_question = "What is the population of Japan?"
    polished2 = generate_polished_answer(unrelated_question, sample_chunks)
    print("\nQuestion:", unrelated_question)
    print("Polished answer:", polished2)