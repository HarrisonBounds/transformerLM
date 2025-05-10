import random

def create_easy_qa_pair(text):
    """
    Creates a simple question-answer pair by taking a sentence as a fact,
    asking a basic "what is X" question, and generating very simple
    distractors.

    Args:
        text (str): A string of text, ideally containing multiple sentences.

    Returns:
        tuple or None: A tuple containing the formatted input string and the
                       correct label (A, B, C, or D), or None if a suitable
                       fact cannot be extracted.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return None

    fact = random.choice(sentences)
    words_in_fact = fact.split()
    if not words_in_fact:
        return None

    # Create a simple stem asking about a random word in the fact
    random_word = random.choice(words_in_fact)
    stem = f"What is '{random_word}' in the following fact?"

    # The correct answer is the word itself
    correct_answer = random_word

    # Generate very simple (and likely poor) distractors
    all_words_in_text = text.split()
    distractors = random.sample([w for w in all_words_in_text if w != correct_answer], k=3)

    all_choices = [correct_answer] + distractors
    random.shuffle(all_choices)
    correct_answer_index = all_choices.index(correct_answer)
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct_label = label_map[correct_answer_index]

    formatted_input = f"[START] {fact} {stem} [A] {all_choices[0]} [B] {all_choices[1]} [C] {all_choices[2]} [D] {all_choices[3]} [ANSWER]"
    return formatted_input, correct_label


text = 'Du Fu ( Wade – Giles : Tu Fu ; Chinese : <unk> ; 712 – 770 ) was a prominent Chinese poet of the Tang dynasty . Along with Li Bai ( Li Po ) , he is frequently called the greatest of the Chinese poets . His greatest ambition was to serve his country as a successful civil servant , but he proved unable to make the necessary accommodations . His life , like the whole country , was devastated by the An Lushan Rebellion of 755 , and his last 15 years were a time of almost constant unrest . Although initially he was little @-@ known to other writers , his works came to be hugely influential in both Chinese and Japanese literary culture . Of his poetic writing , nearly fifteen hundred poems have been preserved over the ages . He has been called the " Poet @-@ Historian " and the " Poet @-@ Sage " by Chinese critics , while the range of his work has allowed him to be introduced to Western readers as " the Chinese Virgil , Horace , Ovid , Shakespeare , Milton , Burns , Wordsworth , Béranger , Hugo or Baudelaire " . '

qa_pair = create_easy_qa_pair(text)

if qa_pair:
    formatted_input, correct_label = qa_pair
    print("Generated Question-Answer Pair:")
    print(f"Input: {formatted_input}")
    print(f"Label: {correct_label}")
else:
    print("Could not generate a question-answer pair from the text.")