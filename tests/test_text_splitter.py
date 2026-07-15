from app.services.text_splitter import TextSplitter


def _build_two_page_fixture():
    """5단어짜리 1페이지 + 5단어짜리 2페이지로 구성된 최소 재현 데이터."""
    page1_words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    page2_words = ["zeta", "eta", "theta", "iota", "kappa"]
    text = " ".join(page1_words) + "\n" + " ".join(page2_words) + "\n"
    page_numbers = [1] * len(page1_words) + [2] * len(page2_words)
    return text, page_numbers


def test_get_page_number_counts_words_not_newlines():
    # page_numbers는 DocumentLoader가 단어 단위로 만들므로, 같은 단위(단어 수)로 조회해야 한다.
    # 개행 문자 수로 세면 실제 위치보다 훨씬 이른 페이지를 잘못 반환하게 된다.
    text, page_numbers = _build_two_page_fixture()
    splitter = TextSplitter()

    pos_in_page1 = text.index("beta")
    pos_in_page2 = text.index("theta")

    assert splitter._get_page_number(text, pos_in_page1, page_numbers) == 1
    assert splitter._get_page_number(text, pos_in_page2, page_numbers) == 2


def test_get_page_number_without_page_numbers_defaults_to_one():
    splitter = TextSplitter()
    assert splitter._get_page_number("아무 텍스트", 0, []) == 1


def test_split_text_assigns_correct_page_to_second_page_chunk():
    text, page_numbers = _build_two_page_fixture()
    # 각 단어가 하나의 청크가 되도록 chunk_size를 작게 설정해 페이지 경계를 명확히 검증한다.
    splitter = TextSplitter(chunk_size=6, chunk_overlap=0)

    chunks = splitter.split_text(text, page_numbers)
    pages = [c["page"] for c in chunks]

    assert pages[0] == 1
    assert pages[-1] == 2
